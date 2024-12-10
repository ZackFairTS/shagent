import pyaudio
import wave
import time
import sys
from threading import Event, Thread
import whisper
import webrtcvad
import numpy as np
import signal
import argparse
from chat_assistant import ChatAssistant


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='语音录制和转录程序')
    parser.add_argument('-t', '--transcribe', action='store_true',
                        help='启用语音转录功能')
    parser.add_argument('-s', '--save-audio', action='store_true',
                        help='保存音频文件')
    parser.add_argument('-w', '--save-text', action='store_true',
                        help='保存转录文本到文件')
    return parser.parse_args()


class VoiceDetector:
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.vad = webrtcvad.Vad(3)  # 设置VAD灵敏度为3（最高）
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)  # 每帧样本数
        self.silent_frames_threshold = 30  # 静音帧阈值
        self.voice_frames_threshold = 5  # 语音帧阈值
        self.silent_frames_count = 0
        self.voice_frames_count = 0
        self.is_speaking = False

    def is_voice(self, audio_chunk):
        """检测音频片段是否包含语音"""
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except:
            return False


class TranscriptionService:
    def __init__(self):
        print("正在加载Whisper模型...")
        self.model = whisper.load_model("tiny")
        print("模型加载完成")

    def transcribe_file(self, audio_file):
        """将音频文件转换为文本"""
        try:
            print("正在转录音频...")
            result = self.model.transcribe(audio_file, language='zh', fp16=False)
            return result["text"].strip()
        except Exception as e:
            raise Exception(f"转录过程中出错: {str(e)}")

    def transcribe_audio_data(self, audio_frames, sample_width=2, channels=1, sample_rate=16000):
        """直接将音频数据转换为文本"""
        try:
            print("正在转录音频...")
            # 将音频帧转换为numpy数组
            audio_data = b''.join(audio_frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            # 标准化音频数据
            audio_np = audio_np / 32768.0

            # 使用Whisper进行转录
            result = self.model.transcribe(audio_np)
            return result["text"].strip()
        except Exception as e:
            raise Exception(f"转录过程中出错: {str(e)}")


class AudioRecorder:
    def __init__(self):
        self.CHUNK = 480  # 30ms at 16kHz
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # 使用16kHz采样率，适合语音识别
        self.recording = False
        self.frames = []
        self.voice_detector = VoiceDetector()
        self.silent_time = 0
        self.speaking_detected = False
        self.silence_threshold = 2.0  # 2秒静音后停止
        self.min_recording_time = 1.0  # 最短录音时间（秒）
        self.recording_start_time = 0
        self.stop_event = Event()
        self.p = None
        self.stream = None

    def initialize_audio(self):
        """初始化音频设备"""
        if self.p is None:
            self.p = pyaudio.PyAudio()

        if self.stream is None:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

    def start_recording(self):
        """开始录音"""
        self.recording = True
        self.frames = []
        self.silent_time = 0
        self.speaking_detected = False
        self.stop_event.clear()

        # 初始化音频设备
        self.initialize_audio()

        print("\n* 正在监听语音...")
        print("* 开始说话时将自动开始录音")
        print("* 停止说话2秒后将自动结束当前段落")
        print("* 按Ctrl+C退出程序")

        # 开始录音线程
        self.audio_thread = Thread(target=self._record)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def _record(self):
        """录音线程"""
        try:
            while not self.stop_event.is_set():
                try:
                    audio_chunk = self.stream.read(self.CHUNK, exception_on_overflow=False)

                    # 检测是否有语音
                    is_voice = self.voice_detector.is_voice(audio_chunk)

                    if is_voice:
                        if not self.speaking_detected:
                            self.speaking_detected = True
                            self.recording_start_time = time.time()
                            print("\n* 检测到语音，开始录音...")
                        self.silent_time = 0
                    else:
                        if self.speaking_detected:
                            self.silent_time += len(audio_chunk) / (self.RATE * self.CHANNELS)

                    # 如果检测到语音或已经开始录音，就保存音频帧
                    if self.speaking_detected:
                        self.frames.append(audio_chunk)

                        # 显示录音状态
                        if is_voice:
                            sys.stdout.write('\r* 正在录音 [说话中]')
                        else:
                            sys.stdout.write('\r* 正在录音 [静音中: {:.1f}s]'.format(self.silent_time))
                        sys.stdout.flush()

                    # 检查是否需要停止录音
                    if self.speaking_detected and self.silent_time >= self.silence_threshold:
                        if time.time() - self.recording_start_time >= self.min_recording_time:
                            print("\n* 检测到语音结束")
                            self.stop_recording()
                            break

                except IOError as e:
                    print(f"\n音频设备错误: {str(e)}")
                    self.stop_recording()
                    break

        except Exception as e:
            print(f"\n录音过程中出错: {str(e)}")
            self.stop_recording()

    def stop_recording(self):
        """停止录音"""
        if not self.recording:
            return

        self.stop_event.set()
        self.recording = False

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"关闭音频流时出错: {str(e)}")

        if hasattr(self, 'p') and self.p is not None:
            try:
                self.p.terminate()
                self.p = None
            except Exception as e:
                print(f"终止PyAudio时出错: {str(e)}")

    def save_recording(self, filename):
        """保存录音为WAV文件"""
        if not self.frames:
            raise Exception("没有录音数据可保存")

        try:
            # 使用上下文管理器保存文件
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.frames))

            print(f"* 录音已保存为: {filename}")

        except Exception as e:
            raise Exception(f"保存录音文件时出错: {str(e)}")


def save_transcription(text, timestamp):
    """保存转录文本到文件"""
    filename = f"transcription_{timestamp}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"* 转录文本已保存为: {filename}")
    except Exception as e:
        raise Exception(f"保存转录文本时出错: {str(e)}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    try:
        # 如果需要转录，预先加载模型
        transcription_service = None
        if args.transcribe:
            transcription_service = TranscriptionService()

        # 创建录音器实例
        recorder = AudioRecorder()
        
        # 创建聊天助手实例
        assistant = ChatAssistant()

        # 注册信号处理函数
        def signal_handler(signum, frame):
            print("\n\n程序正在退出...")
            recorder.stop_recording()
            recorder.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            while True:
                # 开始新一轮录音
                recorder.start_recording()

                # 等待当前录音完成
                while recorder.recording:
                    time.sleep(0.1)

                # 如果需要转录，进行转录
                if args.transcribe:
                    try:
                        if args.save_audio:
                            # 保存音频文件并使用文件进行转录
                            timestamp = int(time.time())
                            filename = f"recording_{timestamp}.wav"
                            recorder.save_recording(filename)
                            text = transcription_service.transcribe_file(filename)
                        else:
                            # 直接使用音频数据进行转录
                            text = transcription_service.transcribe_audio_data(
                                recorder.frames,
                                sample_width=recorder.p.get_sample_size(recorder.FORMAT),
                                channels=recorder.CHANNELS,
                                sample_rate=recorder.RATE
                            )

                            # 送往LLM
                            response = assistant.chat(text)

                        print("\n上一次转录结果:")
                        print("-" * 50)
                        print(text)
                        print("-" * 50)

                        # 如果需要，保存转录文本
                        if args.save_text:
                            timestamp = int(time.time())
                            save_transcription(text, timestamp)

                    except Exception as e:
                        print(f"转录过程中出错: {str(e)}")
                elif args.save_audio:
                    # 如果不需要转录但需要保存音频
                    timestamp = int(time.time())
                    filename = f"recording_{timestamp}.wav"
                    recorder.save_recording(filename)

        except KeyboardInterrupt:
            print("\n\n程序正在退出...")
        finally:
            # 确保录音器被正确清理
            recorder.stop_recording()
            recorder.cleanup()

    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
