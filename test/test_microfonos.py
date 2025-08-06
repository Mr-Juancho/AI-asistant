import pyaudio

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
num_devices = info.get('deviceCount')

print("Dispositivos de audio encontrados:")
print("-" * 30)

for i in range(0, num_devices):
    device_info = p.get_device_info_by_host_api_device_index(0, i)
    if device_info.get('maxInputChannels') > 0:
        print(f"Dispositivo de ENTRADA (Micrófono) ID: {i}")
        print(f"  Nombre: {device_info.get('name')}")
        print(f"  Frecuencia de muestreo por defecto: {device_info.get('defaultSampleRate')}")
        print("-" * 30)

p.terminate()