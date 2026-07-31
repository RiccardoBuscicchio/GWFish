import numpy as np

from GWFish.modules.utilities import (
    early_inspiral_characteristic_strain,
    get_detector_psd,
    get_lgwa_soundcheck_sensitivity,
    monochromatic_dwd_characteristic_strain,
    monochromatic_snr,
    stellar_mass_binary_snr,
)
from GWFish.modules.waveforms import TaylorF2


def test_lgwa_soundcheck_sensitivity_loader():
    psd = get_lgwa_soundcheck_sensitivity()
    ref = get_detector_psd('LGWA_Soundcheck')
    assert psd.shape[1] == 2
    assert np.allclose(psd, ref)


def test_monochromatic_snr_matches_definition():
    frequency = np.array([0.01, 0.03, 0.1])
    characteristic_strain = np.array([1e-20, 2e-20, 3e-20])
    psd_values = np.array([1e-38, 2e-38, 3e-38])
    expected = characteristic_strain / np.sqrt(frequency * psd_values)
    assert np.allclose(monochromatic_snr(characteristic_strain, frequency, psd_values), expected)


def test_monochromatic_dwd_characteristic_strain_scaling():
    frequency = 0.01
    h1 = monochromatic_dwd_characteristic_strain(
        frequency=frequency,
        mass_1=0.6,
        mass_2=0.6,
        luminosity_distance=1.0,
        observation_time=1e7,
    )
    h2 = monochromatic_dwd_characteristic_strain(
        frequency=frequency,
        mass_1=0.6,
        mass_2=0.6,
        luminosity_distance=2.0,
        observation_time=4e7,
    )
    assert np.isclose(h2, h1)


def test_early_inspiral_characteristic_strain_frequency_scaling():
    f1 = 10.
    f2 = 160.
    h1 = early_inspiral_characteristic_strain(f1, 30., 30., 100.)
    h2 = early_inspiral_characteristic_strain(f2, 30., 30., 100.)
    expected_ratio = (f2 / f1) ** (-1 / 6)
    assert np.isclose(h2 / h1, expected_ratio)


def test_stellar_mass_binary_snr_matches_characteristic_strain_integral():
    frequencyvector = np.geomspace(10, 256, 512)
    psd = np.column_stack([frequencyvector, np.full_like(frequencyvector, 1e-46)])
    params = {
        'mass_1': 30.,
        'mass_2': 30.,
        'luminosity_distance': 400.,
        'redshift': 0.,
        'theta_jn': 0.3,
        'phase': 0.,
        'geocent_time': 1187008882.,
    }

    snr = stellar_mass_binary_snr(
        params,
        psd=psd,
        frequencyvector=frequencyvector,
        waveform_class=TaylorF2,
    )

    wave = TaylorF2('TaylorF2', params, {'frequencyvector': frequencyvector, 'f_ref': 50.})()
    h_tilde = np.sqrt(np.sum(np.abs(wave) ** 2, axis=1))
    h_char = 2 * frequencyvector * h_tilde
    h_noise = np.sqrt(frequencyvector * psd[:, 1])
    expected = np.sqrt(np.trapezoid((h_char / h_noise) ** 2, np.log(frequencyvector)))

    assert np.isfinite(snr)
    assert snr > 0
    assert np.isclose(snr, expected)
