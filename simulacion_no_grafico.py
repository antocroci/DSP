import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from typing import Tuple, List, Optional, Dict
import json

class SimuladorDOA:
    """
    Simulador para estimación de dirección de arribo (DOA) usando pyroomacoustics
    """
    
    def __init__(self, fs: int = 16000):
        """
        Inicializa el simulador
        
        Args:
            fs: Frecuencia de muestreo en Hz
        """
        self.fs = fs
        self.c = 343  # Velocidad del sonido en m/s
        self.array_geometry = None
        self.room = None
        self.signals = {}
        
    def crear_array_microfonos(self, 
                              num_mics: int = 4, 
                              spacing: float = 0.1,
                              center_pos: np.ndarray = np.array([2.0, 1.5, 1.0])) -> np.ndarray:
        """
        Crea un array lineal de micrófonos
        
        Args:
            num_mics: Número de micrófonos (default: 4)
            spacing: Separación entre micrófonos en metros (default: 0.1m = 10cm)
            center_pos: Posición del centro del array [x, y, z]
            
        Returns:
            Array con las posiciones de los micrófonos (3 x num_mics)
        """
        # Array lineal en el eje X
        mic_positions = np.zeros((3, num_mics))
        
        # Calcular posiciones simétricas alrededor del centro
        start_offset = -(num_mics - 1) * spacing / 2
        
        for i in range(num_mics):
            mic_positions[0, i] = center_pos[0] + start_offset + i * spacing  # X
            mic_positions[1, i] = center_pos[1]  # Y
            mic_positions[2, i] = center_pos[2]  # Z
            
        self.array_geometry = {
            'positions': mic_positions,
            'num_mics': num_mics,
            'spacing': spacing,
            'center': center_pos
        }
        
        print(f"Array creado: {num_mics} micrófonos, separación {spacing*100:.1f}cm")
        print(f"Posiciones (m):")
        for i in range(num_mics):
            print(f"  Mic {i+1}: [{mic_positions[0,i]:.2f}, {mic_positions[1,i]:.2f}, {mic_positions[2,i]:.2f}]")
            
        return mic_positions
    
    def simular_ambiente_anecoico(self, 
                                 room_size: List[float] = [10, 8, 3],
                                 max_order: int = 0) -> pra.Room:
        """
        Crea un ambiente anecoico (sin reflexiones)
        
        Args:
            room_size: Dimensiones del recinto [largo, ancho, alto] en metros
            max_order: Orden máximo de reflexiones (0 = anecoico)
            
        Returns:
            Objeto Room de pyroomacoustics
        """
        # Crear recinto anecoico
        self.room = pra.ShoeBox(
            room_size,
            fs=self.fs,
            max_order=max_order,  # 0 = sin reflexiones
            absorption=1.0,  # Absorción total
            air_absorption=False
        )
        
        # Agregar array de micrófonos
        if self.array_geometry is None:
            self.crear_array_microfonos()
            
        self.room.add_microphone_array(self.array_geometry['positions'])
        
        print(f"Ambiente anecoico creado: {room_size[0]}x{room_size[1]}x{room_size[2]}m")
        return self.room
    
    def simular_ambiente_reverberante(self, 
                                    room_size: List[float] = [6, 4, 3],
                                    rt60: float = 0.3,
                                    max_order: int = 10) -> pra.Room:
        """
        Crea un ambiente reverberante
        
        Args:
            room_size: Dimensiones del recinto [largo, ancho, alto] en metros
            rt60: Tiempo de reverberación en segundos
            max_order: Orden máximo de reflexiones
            
        Returns:
            Objeto Room de pyroomacoustics
        """
        # Calcular coeficiente de absorción para el RT60 deseado
        volume = np.prod(room_size)
        surface_area = 2 * (room_size[0]*room_size[1] + 
                           room_size[0]*room_size[2] + 
                           room_size[1]*room_size[2])
        
        # Fórmula de Sabine
        absorption = 0.161 * volume / (rt60 * surface_area)
        absorption = min(absorption, 0.99)  # Limitar absorción máxima
        
        # Crear recinto reverberante
        self.room = pra.ShoeBox(
            room_size,
            fs=self.fs,
            max_order=max_order,
            absorption=absorption,
            air_absorption=True
        )
        
        # Agregar array de micrófonos
        if self.array_geometry is None:
            self.crear_array_microfonos()
            
        self.room.add_microphone_array(self.array_geometry['positions'])
        
        print(f"Ambiente reverberante creado: {room_size[0]}x{room_size[1]}x{room_size[2]}m")
        print(f"RT60: {rt60:.2f}s, Absorción: {absorption:.3f}")
        return self.room
    
    def agregar_fuente(self, 
                      signal: np.ndarray,
                      azimuth: float,
                      distance: float = 2.0,
                      elevation: float = 0.0) -> np.ndarray:
        """
        Agrega una fuente sonora al ambiente
        
        Args:
            signal: Señal de audio (1D array)
            azimuth: Ángulo azimutal en grados (0° = frente al array)
            distance: Distancia a la fuente en metros
            elevation: Ángulo de elevación en grados
            
        Returns:
            Posición de la fuente [x, y, z]
        """
        if self.room is None:
            raise ValueError("Debe crear un ambiente primero")
            
        # Convertir ángulos a radianes
        azimuth_rad = np.deg2rad(azimuth)
        elevation_rad = np.deg2rad(elevation)
        
        # Calcular posición de la fuente relativa al centro del array
        center = self.array_geometry['center']
        
        source_pos = np.array([
            center[0] + distance * np.cos(elevation_rad) * np.cos(azimuth_rad),
            center[1] + distance * np.cos(elevation_rad) * np.sin(azimuth_rad),
            center[2] + distance * np.sin(elevation_rad)
        ])
        
        # Agregar fuente al recinto
        self.room.add_source(source_pos, signal=signal)
        
        print(f"Fuente agregada:")
        print(f"  Posición: [{source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f}]m")
        print(f"  Azimuth: {azimuth:.1f}°, Elevación: {elevation:.1f}°, Distancia: {distance:.2f}m")
        
        return source_pos
    
    def simular_propagacion(self, agregar_ruido: bool = False, snr_db: float = 20):
        """
        Simula la propagación acústica y genera las señales en cada micrófono
        
        Args:
            agregar_ruido: Si agregar ruido blanco gaussiano
            snr_db: Relación señal-ruido en dB
        """
        if self.room is None:
            raise ValueError("Debe crear un ambiente y agregar fuentes primero")
            
        # Simular propagación
        self.room.simulate()
        
        # Obtener señales de los micrófonos
        mic_signals = self.room.mic_array.signals
        
        # Agregar ruido si se solicita
        if agregar_ruido:
            for i in range(mic_signals.shape[0]):
                signal_power = np.mean(mic_signals[i, :] ** 2)
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = np.sqrt(noise_power) * np.random.randn(mic_signals.shape[1])
                mic_signals[i, :] += noise
        
        self.signals = {
            'mic_signals': mic_signals,
            'fs': self.fs,
            'num_mics': mic_signals.shape[0],
            'length': mic_signals.shape[1],
            'snr_db': snr_db if agregar_ruido else None
        }
        
        print(f"Simulación completada:")
        print(f"  {mic_signals.shape[0]} micrófonos")
        print(f"  {mic_signals.shape[1]} muestras ({mic_signals.shape[1]/self.fs:.2f}s)")
        if agregar_ruido:
            print(f"  SNR: {snr_db:.1f} dB")
    
    def guardar_senales(self, directorio: str = "simulaciones", 
                       nombre_experimento: str = "exp_001"):
        """
        Guarda las señales simuladas y metadatos
        
        Args:
            directorio: Directorio donde guardar los archivos
            nombre_experimento: Nombre del experimento
        """
        if not self.signals:
            raise ValueError("No hay señales para guardar. Ejecute simular_propagacion() primero")
            
        # Crear directorio si no existe
        os.makedirs(directorio, exist_ok=True)
        
        # Guardar señales individuales
        for i in range(self.signals['num_mics']):
            filename = os.path.join(directorio, f"{nombre_experimento}_mic_{i+1}.wav")
            # Normalizar y convertir a int16
            signal_norm = self.signals['mic_signals'][i, :] / np.max(np.abs(self.signals['mic_signals']))
            signal_int16 = (signal_norm * 32767).astype(np.int16)
            wavfile.write(filename, self.fs, signal_int16)
        
        # Guardar metadatos
        metadata = {
            'experimento': nombre_experimento,
            'fs': self.fs,
            'num_mics': self.signals['num_mics'],
            'array_geometry': {
                'num_mics': self.array_geometry['num_mics'],
                'spacing': self.array_geometry['spacing'],
                'center': self.array_geometry['center'].tolist(),
                'positions': self.array_geometry['positions'].tolist()
            },
            'room_info': {
                'shoebox_dim': self.room.shoebox_dim.tolist(),
                'absorption': getattr(self.room, 'absorption', None),
                'max_order': self.room.max_order
            },
            'snr_db': self.signals.get('snr_db'),
            'length_samples': self.signals['length'],
            'duration_s': self.signals['length'] / self.fs
        }
        
        metadata_file = os.path.join(directorio, f"{nombre_experimento}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Señales guardadas en: {directorio}/")
        print(f"  {self.signals['num_mics']} archivos WAV")
        print(f"  1 archivo de metadatos JSON")
    
    def cargar_senales(self, directorio: str, nombre_experimento: str) -> Dict:
        """
        Carga señales previamente guardadas
        
        Args:
            directorio: Directorio donde están los archivos
            nombre_experimento: Nombre del experimento
            
        Returns:
            Diccionario con señales y metadatos
        """
        # Cargar metadatos
        metadata_file = os.path.join(directorio, f"{nombre_experimento}_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Cargar señales
        num_mics = metadata['num_mics']
        signals = []
        
        for i in range(num_mics):
            filename = os.path.join(directorio, f"{nombre_experimento}_mic_{i+1}.wav")
            fs, signal = wavfile.read(filename)
            signals.append(signal.astype(np.float32) / 32767.0)  # Normalizar
        
        signals_array = np.array(signals)
        
        result = {
            'mic_signals': signals_array,
            'metadata': metadata,
            'fs': fs
        }
        
        print(f"Señales cargadas: {nombre_experimento}")
        print(f"  {num_mics} micrófonos, {signals_array.shape[1]} muestras")
        
        return result
    
    def visualizar_setup(self):
        """
        Visualiza la configuración del experimento
        """
        if self.room is None:
            print("No hay ambiente creado para visualizar")
            return
            
        fig = plt.figure(figsize=(12, 8))
        
        # Vista 3D
        ax1 = fig.add_subplot(121, projection='3d')
        self.room.plot(ax1)
        ax1.set_title('Vista 3D del Setup')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # Vista superior (plano XY)
        ax2 = fig.add_subplot(122)
        
        # Dibujar recinto
        room_corners = np.array([
            [0, 0], [self.room.shoebox_dim[0], 0],
            [self.room.shoebox_dim[0], self.room.shoebox_dim[1]],
            [0, self.room.shoebox_dim[1]], [0, 0]
        ])
        ax2.plot(room_corners[:, 0], room_corners[:, 1], 'k-', linewidth=2, label='Recinto')
        
        # Dibujar micrófonos
        mic_pos = self.array_geometry['positions']
        ax2.scatter(mic_pos[0, :], mic_pos[1, :], c='blue', s=100, marker='o', label='Micrófonos')
        
        # Numerar micrófonos
        for i in range(mic_pos.shape[1]):
            ax2.annotate(f'M{i+1}', (mic_pos[0, i], mic_pos[1, i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Dibujar fuentes
        if hasattr(self.room, 'sources') and self.room.sources:
            for i, source in enumerate(self.room.sources):
                ax2.scatter(source.position[0], source.position[1], 
                           c='red', s=150, marker='*', label=f'Fuente {i+1}' if i == 0 else "")
                ax2.annotate(f'S{i+1}', source.position[:2], 
                            xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Vista Superior (XY)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()

# Ejemplo de uso y pruebas
if __name__ == "__main__":
    # Crear simulador
    sim = SimuladorDOA(fs=16000)
    
    # Crear señal de prueba (chirp)
    duration = 2.0  # segundos
    t = np.linspace(0, duration, int(duration * sim.fs))
    signal = np.sin(2 * np.pi * 1000 * t)  # Tono de 1kHz
    
    print("=== SIMULACIÓN ANECOICA ===")
    # Configurar ambiente anecoico
    sim.simular_ambiente_anecoico()
    
    # Agregar fuente a 30° y 2m de distancia
    sim.agregar_fuente(signal, azimuth=30, distance=2.0)
    
    # Simular con ruido
    sim.simular_propagacion(agregar_ruido=True, snr_db=20)
    
    # Guardar resultados
    sim.guardar_senales(nombre_experimento="anecoico_30deg_2m")
    
    # Visualizar setup
    sim.visualizar_setup()
    
    print("\n=== SIMULACIÓN REVERBERANTE ===")
    # Crear nuevo simulador para ambiente reverberante
    sim2 = SimuladorDOA(fs=16000)
    sim2.simular_ambiente_reverberante(rt60=0.5)
    sim2.agregar_fuente(signal, azimuth=45, distance=1.5)
    sim2.simular_propagacion(agregar_ruido=True, snr_db=15)
    sim2.guardar_senales(nombre_experimento="reverb_45deg_1.5m")
    
    print("\n=== PRUEBA DE CARGA ===")
    # Probar carga de señales
    datos_cargados = sim.cargar_senales("simulaciones", "anecoico_30deg_2m")
    print(f"Forma de las señales cargadas: {datos_cargados['mic_signals'].shape}")