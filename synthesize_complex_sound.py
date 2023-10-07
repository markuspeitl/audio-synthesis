import numpy as np
#sudo apt-get install -y python3-dev libasound2-dev
#https://simpleaudio.readthedocs.io/en/latest/installation.html#installation-ref
import simpleaudio as sa
import scipy

frequency = 220  # Our played note will be 440 Hz
sample_rate = 44100  # 44100 samples per second
seconds = 2  # Note duration of 3 seconds

# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, seconds, seconds * sample_rate, False)





# Generate a 440 Hz sine wave
note = np.sin(frequency * t * 2 * np.pi)

def calc_sin_harmonic(n_degree, amplitude , fundamental_frequency, linear_time_space):
    return np.sin((n_degree + 1) * fundamental_frequency * linear_time_space * 2 * np.pi) * amplitude

def calc_overtone_series(max_degree, fundamental_frequency, linear_time_space):
    harmonic_degrees_array = np.linspace(0, max_degree, max_degree, endpoint=True) + 1

    print(harmonic_degrees_array.shape)

    scaled_time_spaces = np.outer(harmonic_degrees_array, linear_time_space)

    harmonic_waves_array = np.sin(2 * np.pi * fundamental_frequency * scaled_time_spaces)

    print(harmonic_waves_array.shape)

    return harmonic_waves_array



def arange_in_steps(start, end, n_steps):
    
    range_value = end - start 
    step_size = range_value / n_steps
    print(step_size)

    steps_indices = np.linspace(n_steps, 0, n_steps, endpoint=True)

    return (steps_indices / n_steps) * range_value

    return np.arange(start, end, step_size)




max_hearing_frequency = 22000
harmonic_amplitude_falloff = 2
def create_harmonic_sound(fundamental_frequency, linear_time_space):

    last_hearable_harmonic_degree = int(max_hearing_frequency / fundamental_frequency)

    #last_hearable_harmonic_degree = 16

    print("last degree: " + str(last_hearable_harmonic_degree))

    harmonic_waves_array = calc_overtone_series(last_hearable_harmonic_degree, fundamental_frequency, linear_time_space)

    #print(last_hearable_harmonic_degree)
    #harmonic_amplitudes = arange_in_steps(1.0, 0.05, last_hearable_harmonic_degree)
    
    #harmonic_amplitudes = np.linspace(1.0, 0.05, last_hearable_harmonic_degree, endpoint=True)
    #harmonic_amplitudes = np.power(harmonic_amplitudes, 2)

    harmonic_amplitudes = np.linspace(0, last_hearable_harmonic_degree, last_hearable_harmonic_degree, endpoint=True)
    harmonic_amplitudes = 1 / (np.power(harmonic_amplitude_falloff, harmonic_amplitudes))

    

    print(harmonic_amplitudes.shape)

    #np.multiply(a, b[:, np.newaxis])

    scaled_harmonic_waves_array = np.multiply(harmonic_waves_array, harmonic_amplitudes[:, np.newaxis])

    composed_harmonic_signal = np.sum(scaled_harmonic_waves_array, axis=0)

    return composed_harmonic_signal / np.max(composed_harmonic_signal)
    
    #fundamental = calc_sin_harmonic(0, 1.0, fundamental_frequency, linear_time_space)

    
def add_attack_release(sound):
    attack_msec = 500
    attack_samples = int((attack_msec / 1000) * sample_rate)

    release_msec = 2000
    release_samples = (release_msec / 1000) * sample_rate

    start = 0.0
    end = 1.0
    attack_shape = np.arange(start, end, (end - start) / attack_samples)

    attack_shape = np.power(attack_shape, 3)

    print(attack_shape.shape)

    sound[0: 1 * sample_rate] = 0
    sound[1 * sample_rate:1 * sample_rate + attack_samples] = sound[1 * sample_rate:1 * sample_rate + attack_samples] * attack_shape

    sound = sound/ np.max(sound)


def play_sound_buffer(sound_buffers):
    
    
    for sound_buffer in sound_buffers:
        # Ensure that highest value is in 16-bit range
        audio = sound_buffer * (2**15 - 1) / np.max(np.abs(sound_buffer))
        # Convert to 16-bit data
        audio = audio.astype(np.int16)
        
        # Start playback
        play_obj = sa.play_buffer(audio, 1, 2, sample_rate)

        # Wait for playback to finish before exiting
        play_obj.wait_done()


def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def apply_instrument_response_filters(signal):

    signal = lowpass(signal, 2000, sample_rate)

    return signal




"""sounds = [
    note,
    calc_sin_harmonic(1, 0.3, frequency, t),
    calc_sin_harmonic(2, 0.25, frequency, t),
    calc_sin_harmonic(3, 0.2, frequency, t),
    calc_sin_harmonic(4, 0.15, frequency, t),
    calc_sin_harmonic(5, 0.1, frequency, t),
    calc_sin_harmonic(6, 0.1, frequency, t),
    calc_sin_harmonic(7, 0.1, frequency, t),
    calc_sin_harmonic(8, 0.1, frequency, t),
]"""


#sound = np.sum(sounds, axis=0)
#print(sound.shape)
#sound = sound/ np.max(sound)



sound_list = [
    #create_harmonic_sound(55, t) * 0.8,
    #create_harmonic_sound(27, t) * 0.001,
    #create_harmonic_sound(55, t) * 0.001,
    create_harmonic_sound(110, t) * 1.0,
    create_harmonic_sound(220, t) * 0.1,
    create_harmonic_sound(330, t) * 0.01,
    create_harmonic_sound(440, t) * 0.18,
    create_harmonic_sound(550, t) * 0.00002,
    create_harmonic_sound(660, t) * 0.005,
    create_harmonic_sound(770, t) * 0.0001,
    create_harmonic_sound(880, t) * 0.0001,
    create_harmonic_sound(990, t) * 0.001,    
    create_harmonic_sound(1110, t) * 0.0001,
    create_harmonic_sound(1210, t) * 0.0001,
]

composed_signal = np.sum(sound_list, axis=0) / len(sound_list)
#composed_signal = np.power(composed_signal, 2)
composed_signal = composed_signal / np.max(composed_signal)

#sound_list.append(composed_signal)
sound_list = [composed_signal, composed_signal, composed_signal]

combined_sound = np.concatenate(sound_list, axis=0)


#combined_sound = apply_instrument_response_filters(combined_sound)

#add_attack_release(sound)
play_sound_buffer([combined_sound])

#sound = np.concatenate((note, sound))