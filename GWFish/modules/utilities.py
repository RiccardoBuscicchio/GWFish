import GWFish.modules as gw
import GWFish.modules.constants as cst
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parent.parent / 'detectors.yaml'
PSD_PATH = Path(__file__).parent.parent / 'detector_psd'


def get_available_detectors(config=DEFAULT_CONFIG):
    """
    Get the available detectors in the GWFish package, 
    as listed in the .yaml file

    Returns
    -------
    list
        List of available detectors.
    """
    with open(config) as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)
    return doc.keys()

def get_detector_characteristics(detector_name):
    """
    Get the characteristics of a specific detector

    Parameters
    ----------
    detector_name : str
        Name of the detector

    Returns
    -------
    dict
        Dictionary containing the characteristics of the detector
    """
    with open(DEFAULT_CONFIG) as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
    return doc[detector_name]

def get_detector_psd(detector_name):
    """
    Get the Power Spectral Density of a detector

    Parameters
    ----------
    detector_name : str
        Name of the detector

    Returns
    -------
    numpy.ndarray
        Power Spectral Density of the detector
    """
    with open(PSD_PATH / f'{detector_name}_psd.txt', 'rb') as f:
        return np.loadtxt(f, usecols=[0, 1])

def get_lgwa_soundcheck_sensitivity():
    """Return the LGWA soundcheck sensitivity curve as [frequency, PSD]."""
    return get_detector_psd('LGWA_Soundcheck')

def _chirp_mass_from_component_masses(mass_1, mass_2):
    return (mass_1 * mass_2) ** (3 / 5) / (mass_1 + mass_2) ** (1 / 5)

def _evaluate_psd(psd, frequency):
    ff = np.asarray(frequency, dtype=float)
    if callable(psd):
        return np.asarray(psd(ff), dtype=float)

    psd = np.asarray(psd, dtype=float)
    if psd.ndim == 1:
        if psd.shape != ff.shape:
            raise ValueError('If `psd` is 1D it must have the same shape as `frequency`.')
        return psd
    if psd.ndim == 2 and psd.shape[1] >= 2:
        return np.interp(ff, psd[:, 0], psd[:, 1], left=np.inf, right=np.inf)
    raise ValueError('`psd` must be callable, 1D values, or a two-column [frequency, PSD] array.')

def monochromatic_dwd_characteristic_strain(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
    observation_time,
    theta_jn=0.,
):
    """
    Characteristic strain for a monochromatic circular DWD binary.

    `mass_1` and `mass_2` are in solar masses, `luminosity_distance` in Mpc,
    `frequency` in Hz, and `observation_time` in seconds.
    """
    ff = np.asarray(frequency, dtype=float)
    if np.any(ff <= 0):
        raise ValueError('`frequency` must be strictly positive.')
    if observation_time <= 0:
        raise ValueError('`observation_time` must be strictly positive.')

    chirp_mass = _chirp_mass_from_component_masses(mass_1, mass_2) * cst.Msol
    distance = luminosity_distance * cst.Mpc
    if distance <= 0:
        raise ValueError('`luminosity_distance` must be strictly positive.')

    amplitude = (
        2 * (cst.G * chirp_mass) ** (5 / 3) * (np.pi * ff) ** (2 / 3)
        / (cst.c ** 4 * distance)
    )

    cos_iota = np.cos(theta_jn)
    inclination_factor = np.sqrt(((1 + cos_iota ** 2) ** 2) / 4 + cos_iota ** 2)
    monochromatic_strain = amplitude * inclination_factor
    return monochromatic_strain * np.sqrt(ff * observation_time)

def monochromatic_snr(characteristic_strain, frequency, psd):
    """
    Monochromatic SNR from Eq. (2) of arXiv:2002.10462: SNR = h_c / sqrt(f S_n).
    """
    ff = np.asarray(frequency, dtype=float)
    hc = np.asarray(characteristic_strain, dtype=float)
    if np.any(ff <= 0):
        raise ValueError('`frequency` must be strictly positive.')

    sn = _evaluate_psd(psd, ff)
    if np.any(sn <= 0) or np.any(~np.isfinite(sn)):
        raise ValueError('PSD values must be finite and strictly positive at the evaluation frequencies.')

    snr = hc / np.sqrt(ff * sn)
    if np.ndim(snr) == 0:
        return float(snr)
    return snr

def early_inspiral_characteristic_strain(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
):
    """
    Leading-order inspiral characteristic strain h_c(f) = 2 f |h~(f)|.
    """
    ff = np.asarray(frequency, dtype=float)
    if np.any(ff <= 0):
        raise ValueError('`frequency` must be strictly positive.')
    if luminosity_distance <= 0:
        raise ValueError('`luminosity_distance` must be strictly positive.')

    chirp_mass = _chirp_mass_from_component_masses(mass_1, mass_2) * cst.Msol
    distance = luminosity_distance * cst.Mpc

    h_tilde = (
        (cst.c / distance)
        * np.sqrt(5.0 / 24.0)
        * (cst.G * chirp_mass / cst.c ** 3) ** (5.0 / 6.0)
        * np.pi ** (-2.0 / 3.0)
        * ff ** (-7.0 / 6.0)
    )
    return 2 * ff * h_tilde

def stellar_mass_binary_snr(
    source_parameters,
    psd,
    frequencyvector,
    waveform_model='TaylorF2',
    waveform_class=gw.waveforms.LALFD_Waveform,
    f_ref=gw.waveforms.DEFAULT_F_REF,
):
    """
    SNR for stellar-mass binaries from GWFish waveforms and characteristic strain.
    """
    ff = np.asarray(frequencyvector, dtype=float)
    if np.any(ff <= 0):
        raise ValueError('`frequencyvector` must be strictly positive.')

    data_params = {
        'frequencyvector': ff,
        'f_ref': f_ref,
    }
    waveform_obj = waveform_class(waveform_model, source_parameters, data_params)
    polarizations = waveform_obj()
    ff = np.squeeze(waveform_obj.frequencyvector)

    sn = _evaluate_psd(psd, ff)
    valid = np.isfinite(sn) & (sn > 0) & (ff > 0)
    if np.count_nonzero(valid) < 2:
        return 0.

    h_tilde = np.sqrt(np.sum(np.abs(polarizations[valid, :]) ** 2, axis=1))
    h_char = 2 * ff[valid] * h_tilde
    integrand = (h_char ** 2) / (ff[valid] * sn[valid])
    snr_sq = np.trapezoid(integrand, np.log(ff[valid]))
    return float(np.sqrt(max(snr_sq, 0.0)))
    
def add_new_detector(detector_name, dictionary, config=DEFAULT_CONFIG):
    """
    Create a .yaml file from a dictionary

    Parameters
    ----------
    detector_name : str
        Name of the detector
    dictionary : dict
        Dictionary to be saved to the .yaml file
    config : str, optional
    """
    # check all the necessary keys are present
    keys = ['lat', 'lon', 'opening_angle', 'azimuth', 'psd_data', 'duty_factor', 'detector_class',
            'fmin', 'fmax', 'spacing', 'df', 'npoints']
    for key in keys:
        if key not in dictionary.keys():
            raise KeyError(f"Key {key} must be specified in dictionary")
    if 'plot_range' not in list(dictionary.values()):
        dictionary['plot_range'] = '3, 1000, 1e-25, 1e-20'

    new_detector = {detector_name: dictionary}
    with open(config, 'a') as f:
        yaml.dump(new_detector, f, indent=8)


def get_fd_signal(parameters, detector_name, waveform_model, f_ref=gw.waveforms.DEFAULT_F_REF):
    """
    Get the frequency domain signal projected onto the detector from a waveform model 
    and a set of parameters

    Parameters
    ----------
    parameters : pandas.DataFrame
        DataFrame containing the parameters of the event
    network : gw.DetectorNetwork

    waveform_model : str

    Returns
    -------
    numpy.ndarray
        Signal projected onto the detector
    """

    # The waveform model can be accessed through the waveform_class attribute,
    # which requires the waveform_model and the data_params and the parameters of the event
    detector = gw.detection.Detector(detector_name)
    waveform_class = gw.waveforms.LALFD_Waveform
    data_params = {
            'frequencyvector': detector.frequencyvector,
            'f_ref': f_ref
        }
    waveform_obj = waveform_class(waveform_model, parameters.iloc[0], data_params)
    wave = waveform_obj()
    t_of_f = waveform_obj.t_of_f

    # The waveform is then projected onto the detector taking into account the Earth rotation 
    # by passing at each frequency step the time of the waveform at the detector
    signal = gw.detection.projection(parameters.iloc[0], detector, wave, t_of_f)

    return signal, t_of_f


def get_snr(parameters, network, waveform_model, f_ref=gw.waveforms.DEFAULT_F_REF):
    """
    Get the Signal-to-Noise Ratio of single detectors and combined in a network

    Parameters
    ----------
    parameters : pandas.DataFrame
        DataFrame containing the parameters of the event
    network : gw.DetectorNetwork
        Detector network
    waveform_model : str
        Waveform model

    Returns
    -------
    pandas.DataFrame
        Signal-to-Noise Ratio in individual detectors and in the network
    """
    waveform_class = gw.waveforms.LALFD_Waveform

    nsignals = len(parameters)
    
    # The SNR is then computed by taking the norm of the signal projected onto the detector
    # and dividing by the noise of the detector
    snrs = {}
    for i in range(nsignals):
        snr = {}
        for detector in network.detectors:
            data_params = {
                'frequencyvector': detector.frequencyvector,
                'f_ref': f_ref
            }
            waveform_obj = waveform_class(waveform_model, parameters.iloc[i], data_params)
            wave = waveform_obj()
            t_of_f = waveform_obj.t_of_f
            signal = gw.detection.projection(parameters.iloc[i], detector, wave, t_of_f)

            snr[detector.name] = np.sqrt(np.sum(gw.detection.SNR(detector, signal)**2))

        snr['network'] = np.sqrt(np.sum([snr[detector.name]**2 for detector in network.detectors]))
        snrs['event_' + str(i)] = snr

    return pd.DataFrame.from_dict(snrs, orient='index')

 