import autorootcwd
import cv2
import time
import os
import numpy as np
import threading
import logging
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO
from demo.core.stream import StreamManager
from demo.core.detector import DetectionProcessor
from demo.services.cada import CADAService
from demo.services.mqtt import MQTTService
from demo.services.mqtt_publisher import MQTTPublisher
from demo.services.mqtt_manager import MQTTManager
from demo.services.ptz import PTZService
from demo.utils.alerts import AlertManager, AlertCodes
from demo.config.settings import HOST, PORT, DEBUG
from demo.config.settings import BROKER_ADDR, BROKER_PORT, MQTT_PTZ_TOPIC, MQTT_PTZ_CLIENT_ID
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
FFMPEG_OPTS = (
    "fflags nobuffer;"
    "flags low_delay;"
    "probesize 32;"
    "analyzeduration 0"  
)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FFMPEG_OPTS

logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

class HumanDetectionApp:
    def __init__(self):
        self.app = Flask(__name__,
                        template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
        self.socketio = SocketIO(async_mode="threading", logger=False, engineio_logger=False)
        self.socketio.init_app(self.app)
        self.stream_manager = StreamManager()
        self.detection_processor = DetectionProcessor()
        self.cada_service = CADAService(self.socketio)

        self.mqtt_mgr = MQTTManager()
        self.mqtt_service = MQTTService(self.stream_manager, self.detection_processor)
        self.ptz_service = PTZService()
        
        self.detection_processor.ptz_service = self.ptz_service

        self.ptz_service.initialize(640, 480) 
        self.ptz_initialized = True


        threading.Thread(target=self._heavy_init, daemon=True).start()
        self._start_services()
        self._setup_routes()
        self._register_socketio_handlers()
        self.ptz_service.reset()
    def _heavy_init(self):
        print("[INIT] models pre-loaded")
    def _start_services(self):
        self.cada_service.start()
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')
        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self.get_stream_generator(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        @self.app.route('/alerts')
        def alerts():
            def event_stream():
                self.detection_processor.alert_manager.send_alert(AlertCodes.SYSTEM_STARTED, "시스템 시작")
                
                while True:
                    try:
                        data = self.detection_processor.alert_manager.get_next_alert(timeout=1.0)  # 타임아웃 증가
                        if data:
                            yield f"data: {data}\n\n"
                        else:
                            yield "data: \n\n"
                    except Exception as e:
                        print(f"[SSE ERROR] {e}")
                        time.sleep(0.1)
                        continue
                        
            return Response(
                event_stream(), 
                mimetype='text/event-stream',           headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Cache-Control'
                }
            )
        @self.app.route('/reset', methods=['POST'])
        def reset():
            self.stream_manager.stop_stream()
            self.mqtt_service._send_stream_off()
            self.ptz_service.reset()
            return jsonify({'success': True})
            
        @self.app.route('/redetection', methods=['POST'])
        def redetection():
            self.detection_processor.force_redetection()
            return jsonify({'success': True})
        @self.app.route('/timestamp')
        def timestamp():
            return jsonify({'timestamp': self.get_last_timestamp()})
        @self.app.route('/analysis_result', methods=['POST'])
        def analysis_result():
            try:
                data = request.get_json()
                if not data:
                    return jsonify({
                      'status': 'error',
                    'message': 'No JSON data provided'
                    }), 400
                
                description = data.get('description', '')
                bbox_normalized = data.get('bbox_normalized', [])
                signal_type = data.get('signal_type', 'analysis')
                
                if description:
                    message = f"이상행동 분석 결과 - {description}"
                    print(f"[DAM] Received analysis: {description}")
                    
                    self.detection_processor.alert_manager.send_alert(
                        AlertCodes.INTRUSION_DETECTED,
                        message
                    )
                    
                    return jsonify({
                        'status': 'success',
                  'message': 'Analysis result received and sent to web interface'
                    }), 200
                else:
                    print("[DAM] No description in analysis result")
                    return jsonify({
                      'status': 'error',
                    'message': 'No description provided'
                    }), 400
            except Exception as e:
                print(f"[WEB ERROR] Analysis result: {e}")
                return jsonify({
                  'status': 'error',
                    'message': str(e)
                }), 500

        @self.app.route('/zone_status', methods=['POST'])
        def zone_status():
            try:
                zone_data = request.get_json()
                #print(f"[WEB] Zone status received: {zone_data}")
                self.socketio.emit('zone_status', zone_data, namespace='/csi')
                
                return jsonify({'status': 'success'}), 200
            except Exception as e:
                #print(f"[WEB ERROR] Zone status: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
            
        @self.app.route('/mode_status', methods=['POST'])
        def mode_status():
            try:
                mode_data = request.get_json()
                #print(f"[WEB] Mode status received: {mode_data}")
                self.socketio.emit('mode_status', mode_data, namespace='/csi')
                
                return jsonify({'status': 'success'}), 200
            except Exception as e:
                #print(f"[WEB ERROR] Mode status: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
            
    def _register_socketio_handlers(self):
        @self.socketio.on("connect", namespace="/csi")
        def on_connect():
            if self.cada_service.mqtt_manager:
                self.cada_service.mqtt_manager.start()
            print("[SocketIO] Client connected")
        @self.socketio.on("disconnect", namespace="/csi")
        def on_disconnect():
            print("[SocketIO] Client disconnected")

    def process_frame(self, frame):
        if frame is None:
            return None
        processed_frame, bbox_for_ptz = self.detection_processor.process_frame(frame)
        if self.ptz_initialized:
            self.ptz_service.update(bbox_for_ptz)
        return processed_frame
    
    def gen_frames(self):
        while True:
            if not self.stream_manager.is_active():
                blank = self.stream_manager.get_blank_frame()
                ok, buf = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue
            frame = self.stream_manager.get_frame()
            if frame is None:
                blank = self.stream_manager.get_blank_frame()
                ok, buf = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                    buf.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue
            processed_frame = self.process_frame(frame)
            if processed_frame is not None:
                ok, buf = cv2.imencode('.jpg', processed_frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ok:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                           buf.tobytes() + b'\r\n')

    def get_stream_generator(self):
        return self.gen_frames()
    def get_last_timestamp(self):
        return self.detection_processor.last_timestamp
    def run(self):
        self.app.run(host=HOST, port=PORT, debug=DEBUG)
if __name__ == "__main__":
    app = HumanDetectionApp()
    app.run()



    