"""
Beamforming a WERA .SORT file into directional Doppler spectra and radial currents.

Pipeline (book: Guerin, "High-Frequency Radars for the Oceanographer"):

    0. read the .SORT file: 512-byte text header + range-sorted complex
       time series, shape (chirps x ranges x antennas)          (sec. 11.1)
    1. read the radar parameters from the header and the antenna positions
       from a site file                                         (sec. 2.2, 2.3)
    2. steering vector for an array of ANY shape                (sec. 4.2, eq 4.2.15)
    3. taper (Hann) weights across the array                    (sec. 4.4)
    4. beam pattern of this particular array                    (sec. 4.1-4.4)
    5. beamform every range cell over a fan of bearings         (sec. 4.2, 4.5)
    6. Doppler spectra, averaged over overlapping sub-windows   (sec. 3.3, 3.4)
    7. radial current from the Bragg peak shifts                (sec. 2.6, 11.2)

Range processing (the FFT over fast time) is NOT done here: the radar has
already done it before writing the .SORT file (sec. 3.2, 11.1).

Why a site file? A .SORT header stores the number of antennas, the carrier
frequency, the chirp period, the range cell size, the station position and
a per-channel phase-correction table, but NOT where the antennas stand.
Their positions come from the site survey / WERA site configuration and have
to be supplied separately (see SITE FILE below).

File layout follows the open-source matWERA reader (G. Voulgaris, Univ. of
South Carolina, read_WERA_sort.m / read_WERA_header.m):
    - 512 bytes of ASCII header, fixed column positions
    - then, for each range cell, for each antenna: NSER chirps of
      interleaved float32 (I, Q), little-endian.

SITE FILE (JSON). Antenna positions in metres, east and north of any
reference point, listed in the same order as the channels in the .SORT file:

    {
      "site": "MYSITE",
      "antennas_en_m": [[0.0, 0.0], [9.1, -1.2], ...],
      "boresight_deg": 160.0,          # compass bearing the array looks toward (sea)
      "scan_halfwidth_deg": 60,        # optional, default 60 (book p. 57)
      "phase_cal_deg": [0, 12.5, ...], # optional per-channel phase error to remove
      "steering_sign": 1               # optional, +1 or -1 (see steering_vector)
    }

Run:
    python beamforming_sort.py FILE.SORT SITE.json [--range-km 20] [--bearing 160]
"""

import argparse
import json
import re
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")              # render to file, no interactive window needed
import matplotlib.pyplot as plt


C0 = 299_792_458.0                 # speed of light [m/s]
G = 9.81                           # gravity [m/s^2] — sets the Bragg wave speed
HEADER_BYTES = 512


# ----------------------------------------------------------------------------
# 0. READ THE .SORT FILE
# ----------------------------------------------------------------------------

def _num(text, default=None):
    """Parse a number out of a fixed-width header field; `default` if blank."""
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(m.group()) if m else default


def parse_sort_header(raw):
    """Decode the 512-byte WERA header into a dict.

    Column positions are those used by matWERA (converted to 0-based Python
    slices). Only the fields needed for beamforming are kept.
    """
    h = raw.decode("latin-1")
    hdr = {"code": h[48:54]}
    if hdr["code"] not in ("FMSOCO", "FMSOCN"):
        raise ValueError(f"header code {hdr['code']!r}: not a SORT/RFI file")

    ix = h.find("SAMPLES")
    hdr["n_chirps"] = int(_num(h[:ix]))                 # slow-time samples
    hdr["n_ant"] = int(_num(h[223:225]))                # receive channels
    hdr["n_ranges"] = int(_num(h[150:153]))             # range cells
    hdr["freq_hz"] = _num(h[64:71]) * 1e6               # carrier [MHz -> Hz]
    hdr["t_chirp_header"] = _num(h[131:139])            # chirp period [s]

    # Range cell size [km]; older headers store it in another place.
    if h[89:94] == "RANGE":
        hdr["range_res_km"] = _num(h[96:103])
    else:
        hdr["range_res_km"] = _num(h[100:102]) * 0.15 * 8

    # Station position (deg, min, sec, hemisphere).
    lon = _num(h[169:172], 0) + _num(h[173:175], 0) / 60 + _num(h[176:178], 0) / 3600
    lat = _num(h[187:190], 0) + _num(h[191:193], 0) / 60 + _num(h[194:196], 0) / 3600
    hdr["lon"] = -lon if h[179] == "W" else lon
    hdr["lat"] = -lat if h[197] == "S" else lat

    hdr["site"] = h[36:48].strip()
    m = re.search(r"TRUENORTH:\s*([-+]?\d+\.?\d*)", h)
    hdr["truenorth_deg"] = float(m.group(1)) if m else None   # array orientation
    try:
        hdr["time"] = datetime.strptime(
            f"{h[15:17]} {h[18:21]} {h[22:24]} {h[25:27]}:{h[28:30]}",
            "%d %b %y %H:%M")
    except ValueError:
        hdr["time"] = None

    hdr["mt"] = int(_num(h[202:208])) if h[199:202] == "MT:" else 1536
    hdr["mode"] = int(h[229:232].strip() or "0", 16) if h[226:229] == "MD:" else 0
    hdr["rxoffset"] = _num(h[253:263], 0.0) if h[244:253] == "RXOFFSET:" else 0.0

    # Range offset of the first cell, in cells.
    if h[238] != "T":
        hdr["ran_off"] = _num(h[238:244], 0.0)
    elif h[239] != ":":
        hdr["ran_off"] = _num(h[239:244], 0.0)
    else:
        hdr["ran_off"] = _num(h[240:244], 0.0)

    # Per-channel phase-correction table: 16 fields of 8 characters. Its unit
    # is not documented in matWERA, so it is reported but not applied.
    hdr["phase_cor_header"] = [_num(h[384 + 8 * i: 392 + 8 * i]) for i in range(16)]

    hdr["t_chirp"] = exact_chirp_period(hdr)
    return hdr


def exact_chirp_period(hdr):
    """The header rounds the chirp period; rebuild it from the ADC clock.

    Same logic as matWERA: MT samples per chirp at a clock set by bit 3 of
    the mode word; if that disagrees with the header by >1 % the radar ran
    in VHF mode with a doubled ADC clock.
    """
    if hdr["mode"] & 8:
        adc_clk = 180.0e6 / (4.0 * 7618.0)
    else:
        adc_clk = 181.48430768e6 / (4.0 * 7680.0)
    t = hdr["mt"] / adc_clk
    if abs(t - hdr["t_chirp_header"]) > 0.01 * hdr["t_chirp_header"]:
        t = hdr["mt"] / (2 * adc_clk)
    if abs(t - hdr["t_chirp_header"]) > 0.01 * hdr["t_chirp_header"]:
        t = hdr["t_chirp_header"]                    # fall back to the header
    return t


def read_sort(path):
    """Return (header dict, data) with data[chirp, range, antenna] complex64."""
    with open(path, "rb") as f:
        hdr = parse_sort_header(f.read(HEADER_BYTES))
    n_t, n_r, n_a = hdr["n_chirps"], hdr["n_ranges"], hdr["n_ant"]
    raw = np.fromfile(path, dtype="<f4", offset=HEADER_BYTES,
                      count=2 * n_t * n_r * n_a)
    if raw.size != 2 * n_t * n_r * n_a:
        raise ValueError(f"file holds {raw.size} floats, header promises "
                         f"{2 * n_t * n_r * n_a}")
    iq = raw.reshape(n_r, n_a, n_t, 2)               # range, antenna, chirp, I/Q
    data = (iq[..., 0] + 1j * iq[..., 1]).astype(np.complex64)
    data = np.transpose(data, (2, 0, 1))             # -> chirp, range, antenna

    # Distance to the centre of each range cell [km] (matWERA convention).
    res = hdr["range_res_km"]
    hdr["ranges_km"] = (np.arange(n_r) + 0.5) * res - hdr["ran_off"] * res
    return hdr, data


# ----------------------------------------------------------------------------
# 1. RADAR PARAMETERS (from the header) AND ARRAY GEOMETRY (from the site file)
# ----------------------------------------------------------------------------

class Radar:
    """Everything derived from the header and the site file, in one place."""

    def __init__(self, hdr, site):
        self.hdr = hdr
        self.f0 = hdr["freq_hz"]
        self.lam = C0 / self.f0                       # radar wavelength (eq 2.2.10)
        self.k0 = 2 * np.pi / self.lam                # EM wavenumber
        self.f_bragg = np.sqrt(G / (np.pi * self.lam))  # eq 2.3.18
        self.pri = hdr["t_chirp"]                     # slow-time step [s]

        pos = np.asarray(site["antennas_en_m"], dtype=float)
        if pos.shape != (hdr["n_ant"], 2):
            raise ValueError(f"site file lists {len(pos)} antennas, "
                             f"SORT file has {hdr['n_ant']}")
        self.pos = pos - pos.mean(axis=0)             # centre the array
        self.boresight = float(site["boresight_deg"])
        half = float(site.get("scan_halfwidth_deg", 60))
        self.bearings = np.arange(self.boresight - half,
                                  self.boresight + half + 1e-9, 1.0)
        self.sign = int(site.get("steering_sign", 1))
        cal = site.get("phase_cal_deg")
        self.cal = (np.exp(-1j * np.radians(np.asarray(cal, float)))
                    if cal is not None else np.ones(hdr["n_ant"]))

        # Coordinate of each antenna along the array's long axis (perpendicular
        # to boresight), used for the taper and for the aperture.
        b = np.radians(self.boresight)
        axis = np.array([np.cos(b), -np.sin(b)])      # boresight rotated 90 deg
        self.s = self.pos @ axis

    def report(self):
        h = self.hdr
        n = h["n_ant"]
        aperture = np.ptp(self.s)
        order = np.argsort(self.s)
        gaps = np.linalg.norm(np.diff(self.pos[order], axis=0), axis=1)
        print(f"site / time           : {h['site']}  {h['time']}")
        print(f"station               : {h['lat']:.4f} N, {h['lon']:.4f} E")
        print(f"cube (chirps,rng,ant) : {h['n_chirps']} x {h['n_ranges']} x {n}")
        print(f"radar frequency       : {self.f0 / 1e6:8.3f} MHz")
        print(f"radar wavelength      : {self.lam:8.2f} m")
        print(f"Bragg frequency       : {self.f_bragg:8.3f} Hz")
        print(f"chirp period          : {self.pri:8.4f} s  "
              f"(Doppler band +/- {1 / (2 * self.pri):.2f} Hz)")
        print(f"range cell            : {h['range_res_km']:8.3f} km")
        print(f"array aperture        : {aperture:8.1f} m")
        print(f"antenna gaps          : {gaps.min():.2f} - {gaps.max():.2f} m "
              f"(lambda/2 = {self.lam / 2:.2f} m)")
        # Width of the main lobe at boresight, eq 4.1.5 with L ~ N*d.
        print(f"beamwidth ~0.89 lam/L : "
              f"{np.degrees(0.89 * self.lam / (aperture * n / (n - 1))):6.1f} deg "
              f"(uniform; ~1.5x wider with Hann)")
        print(f"bearings scanned      : {self.bearings[0]:.0f} to "
              f"{self.bearings[-1]:.0f} deg true")
        # A grating lobe enters the scanned fan once d > lam / (1 + sin(theta_max)).
        half = np.radians(np.ptp(self.bearings) / 2)
        if gaps.max() > self.lam / (1 + np.sin(half)):
            print(f"WARNING: largest antenna gap {gaps.max():.2f} m exceeds "
                  f"{self.lam / (1 + np.sin(half)):.2f} m -> weak grating lobes "
                  f"possible at the edge of the scan (sec. 4.2, eq 4.2.9)")
        if n < 6:
            print("WARNING: fewer than 6 antennas: beamforming resolution is poor; "
                  "the book uses direction finding for such arrays (Ch. 5)")
        if h["rxoffset"]:
            print("NOTE: RXOFFSET is non-zero (bistatic/slave receiver); its "
                  "Doppler offset is not corrected here")


# ----------------------------------------------------------------------------
# 2. THE STEERING VECTOR — for an array of any shape
# ----------------------------------------------------------------------------

def steering_vector(radar, bearing_deg):
    """Phase each antenna sees for an echo arriving from compass `bearing_deg`.

    The straight-line formula exp(i n k d sin(theta)) only holds for equally
    spaced antennas on a perfect line. Real sites rarely are: the coastline
    bends the array, spacings are adjusted to the terrain, and some arrays
    are 2-D. The book's general form (eq 4.2.14-4.2.15) uses each antenna's
    position vector d_n and the unit vector u pointing toward the sea patch:

        a_n(phi) = exp(i * k0 * u(phi) . d_n)

    For a straight, evenly spaced line this reduces to the old formula.

    `steering_sign` flips the phase convention. Whether +1 or -1 is right
    depends on the receiver's I/Q convention; with the wrong sign every
    bearing is mirrored about boresight. Check it once on a known target
    (a ship, an island, a point of land) and fix it in the site file.

    Returns shape (n_bearings, n_ant).
    """
    b = np.radians(np.atleast_1d(bearing_deg))
    u = np.stack([np.sin(b), np.cos(b)], axis=1)      # (east, north) unit vectors
    return np.exp(1j * radar.sign * radar.k0 * (u @ radar.pos.T))


# ----------------------------------------------------------------------------
# 3. TAPER WINDOWS — the sidelobe / resolution trade-off
# ----------------------------------------------------------------------------

def taper(radar, kind="hann"):
    """Weight per antenna before summing (sec. 4.4).

    The book's Hann weights W(n) = sin^2(pi n / (N+1)) assume evenly spaced
    antennas numbered 1..N. Here n is replaced by each antenna's actual
    position along the array, so the fade-out follows the real geometry and
    falls back to the book's formula for a regular array.
    """
    n_ant = len(radar.s)
    if kind == "uniform":
        return np.ones(n_ant)
    if kind == "hann":
        frac = (radar.s - radar.s.min()) / np.ptp(radar.s)   # 0 .. 1 along array
        n_equiv = 1 + frac * (n_ant - 1)                     # 1 .. N
        return np.sin(np.pi * n_equiv / (n_ant + 1)) ** 2
    raise ValueError(f"unknown taper: {kind!r}")


# ----------------------------------------------------------------------------
# 4. BEAM PATTERN — of THIS array, steered at boresight
# ----------------------------------------------------------------------------

def beam_pattern(radar, steer_deg, look_deg, taper_kind="hann"):
    """Array factor (eq 4.4.22) in dB: weights for `steer_deg`, echoes from `look_deg`."""
    w = taper(radar, taper_kind) * steering_vector(radar, steer_deg)[0]
    a = steering_vector(radar, look_deg)                     # (n_look, n_ant)
    mag = np.abs(a @ np.conj(w)) / np.sum(taper(radar, taper_kind))
    return 20 * np.log10(np.maximum(mag, 1e-6))


# ----------------------------------------------------------------------------
# 5. AZIMUTH PROCESSING — delay and sum, every range, every bearing
# ----------------------------------------------------------------------------

def beamform(radar, data, taper_kind="hann"):
    """data (chirps, ranges, antennas) -> (chirps, ranges, bearings), eq 4.2.16.

    Channel phase errors from the site file are removed first, so that the
    antennas line up as the geometry predicts.
    """
    w_taper = taper(radar, taper_kind)
    W = (w_taper[:, None] * np.conj(steering_vector(radar, radar.bearings)).T)
    W = (W / np.sum(w_taper)).astype(np.complex64)            # (n_ant, n_bearings)
    return (data * radar.cal.astype(np.complex64)) @ W


# ----------------------------------------------------------------------------
# 6. DOPPLER PROCESSING — FFT along slow time, averaged over sub-windows
# ----------------------------------------------------------------------------

def doppler_spectrum(radar, beamed, n_fft=512, overlap=0.5):
    """Power spectra along axis 0, averaged over overlapping Hann windows.

    A single long FFT of real sea echo is very noisy: each spectral value
    scatters as much as its own mean (eq 3.3.20). Splitting the record into
    overlapping windows and averaging the sub-spectra reduces that scatter by
    the number of windows (eq 3.3.18-3.3.19), at the cost of resolution 1/T
    per window. The book uses ~10-30 sub-spectra (p. 37).
    """
    n_t = beamed.shape[0]
    n_fft = min(n_fft, n_t)
    step = max(1, int(n_fft * (1 - overlap)))
    starts = range(0, n_t - n_fft + 1, step)
    win = np.hanning(n_fft).reshape((-1,) + (1,) * (beamed.ndim - 1))

    power = 0.0
    for k, s in enumerate(starts, 1):
        seg = beamed[s:s + n_fft] * win
        power = power + np.abs(np.fft.fft(seg, axis=0)) ** 2
    power = np.fft.fftshift(power / k, axes=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=radar.pri))
    return freqs, power, k


# ----------------------------------------------------------------------------
# 7. RADIAL CURRENT from the Bragg peaks
# ----------------------------------------------------------------------------

def _peak(freqs, spec, centre, half_width):
    """Strongest bin within centre +/- half_width, refined to sub-bin accuracy."""
    sel = np.flatnonzero(np.abs(freqs - centre) <= half_width)
    i = sel[np.argmax(spec[sel])]
    f = freqs[i]
    if 0 < i < len(spec) - 1:                       # parabola through 3 bins (dB)
        y0, y1, y2 = 10 * np.log10(spec[i - 1:i + 2] + 1e-30)
        den = y0 - 2 * y1 + y2
        if den < 0:
            f += 0.5 * (y0 - y2) / den * (freqs[1] - freqs[0])
    return f, spec[i]


def radial_current(radar, freqs, power, max_current=1.5, min_snr_db=10.0):
    """Radial current [m/s, + = away from radar] for every spectrum in `power`.

    Each Bragg line is displaced by the same f_U (eq 11.2.3); f_U = -2 v / lam
    (eq 2.2.10). Both lines are used when both stand min_snr_db above the
    noise floor, otherwise only the stronger one. NaN where neither does.
    """
    lam, fb = radar.lam, radar.f_bragg
    half = min(2 * max_current / lam, 0.5 * fb)      # search window around +/- fB
    flat = power.reshape(power.shape[0], -1)
    v = np.full(flat.shape[1], np.nan)
    snr = np.full(flat.shape[1], np.nan)

    # Noise floor: far from both Bragg lines and from zero Doppler.
    far = np.abs(freqs) > fb + half + 0.1
    if far.sum() < 8:
        far = np.ones_like(freqs, bool)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        noise = np.median(col[far])
        fp, pp = _peak(freqs, col, +fb, half)
        fn, pn = _peak(freqs, col, -fb, half)
        snr_p, snr_n = 10 * np.log10(pp / noise), 10 * np.log10(pn / noise)
        shifts = [f - c for f, c, s in ((fp, fb, snr_p), (fn, -fb, snr_n))
                  if s >= min_snr_db]
        if shifts:
            v[j] = -np.mean(shifts) * lam / 2
        snr[j] = max(snr_p, snr_n)
    return v.reshape(power.shape[1:]), snr.reshape(power.shape[1:])


def noise_floor(radar, freqs, power):
    """Noise power per range cell: median over bearings of the far Doppler bins.

    Beyond ~2 fB the sea echo has died away (sec. 11.3: "averaging the tail of
    the spectra at large Doppler frequencies"); the median ignores the odd
    ship or interference spike there.
    """
    far = np.abs(freqs) > 2 * radar.f_bragg
    if far.sum() < 8:
        far = np.abs(freqs) > 1.5 * radar.f_bragg
    return np.median(power[far], axis=(0, 2))


# ----------------------------------------------------------------------------
# 9. OBJECT DETECTION — ships and fixed targets
# ----------------------------------------------------------------------------

def detect_objects(radar, beamed, rng, n_fft=128, overlap=0.5,
                   min_snr_db=15.0, min_range_km=3.0, max_current=1.0):
    """Find echoes that are not sea echo, in every range x bearing cell.

    The sea always sits in the same places of the Doppler spectrum: the two
    Bragg lines at +/- fB, shifted by at most a few tenths of a Hz by the
    current (sec. 2.6). Anything bright elsewhere is something else:

    - moving objects (ships) give an isolated spot at the Doppler of their own
      radial speed, v = -f lam/2 (sec. 11.4: "Ships can also be seen in the
      range-Doppler map as isolated spots");
    - fixed objects (platforms, buoys, anchored ships, land) give a spot at
      zero Doppler (sec. 11.3);
    - a ship crossing a cell is only there for part of the record, so the
      MEAN of the short spectra jumps while their MEDIAN does not (sec. 11.5,
      Fig. 11.10). mean / median is therefore a "something passed by" score.

    Short windows suit ships: they move between cells within minutes.

    Returns a dict of range x bearing maps and the Doppler axis.
    """
    n_t = beamed.shape[0]
    n_fft = min(n_fft, n_t)
    step = max(1, int(n_fft * (1 - overlap)))
    win = np.hanning(n_fft)[:, None, None]
    segs = [np.abs(np.fft.fft(beamed[s:s + n_fft] * win, axis=0)) ** 2
            for s in range(0, n_t - n_fft + 1, step)]
    segs = np.fft.fftshift(np.stack(segs), axes=1)    # (window, f, range, bearing)
    mean, median = segs.mean(axis=0), np.median(segs, axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=radar.pri))
    df = freqs[1] - freqs[0]

    noise = noise_floor(radar, freqs, mean)[None, :, None]
    snr = 10 * np.log10(mean / noise)

    zero = np.abs(freqs) <= 1.5 * df                       # fixed objects
    sea_hw = 2 * max_current / radar.lam + 1.5 * df         # Bragg +/- current
    sea = np.abs(np.abs(freqs) - radar.f_bragg) <= sea_hw
    moving = ~zero & ~sea

    i_mov = np.argmax(np.where(moving[:, None, None], snr, -np.inf), axis=0)
    mov_snr = np.take_along_axis(snr, i_mov[None], 0)[0]
    static_snr = snr[zero].max(axis=0)
    ratio = 10 * np.log10(mean / median)
    trans = np.where(moving[:, None, None], ratio, -np.inf).max(axis=0)

    near = rng < min_range_km                              # the site's own clutter
    for m in (mov_snr, static_snr, trans):
        m[near] = np.nan
    return dict(freqs=freqs, snr=snr, n_windows=len(segs), n_fft=n_fft,
                moving_snr=mov_snr, moving_hz=freqs[i_mov],
                static_snr=static_snr, transient_db=trans,
                min_snr_db=min_snr_db, sea_hw=sea_hw, rng=rng)


def _local_peaks(score, min_db, d_range=1, d_bearing=6):
    """Cells above min_db that are the maximum of their neighbourhood.

    One object lights up a few neighbouring bearings (the beam is ~12 deg
    wide, sec. 4.5) and sometimes the next range cell; keep only the peak.
    """
    out = []
    s = np.nan_to_num(score, nan=-np.inf)
    for i, j in zip(*np.nonzero(s >= min_db)):
        blk = s[max(0, i - d_range):i + d_range + 1, max(0, j - d_bearing):j + d_bearing + 1]
        if s[i, j] >= blk.max():
            out.append((i, j))
    return out


def _latlon(radar, r_km, brg_deg):
    """Position of a cell, flat-earth approximation (fine within ~150 km)."""
    lat0, lon0 = radar.hdr["lat"], radar.hdr["lon"]
    dn = r_km * np.cos(np.radians(brg_deg))
    de = r_km * np.sin(np.radians(brg_deg))
    return lat0 + dn / 111.32, lon0 + de / (111.32 * np.cos(np.radians(lat0)))


def mains_ranges_km(radar, max_km, mains_hz=50.0):
    """Ranges where power-line hum lands after FMCW range processing.

    A WERA turns range into a beat frequency: one range cell (dR) per 1/T Hz,
    so a tone at k * mains_hz shows up at k * mains_hz * T * dR. The book
    notes these "bright spots on the zero-Doppler line ... are an artifact of
    the power grid impurity that generates extra 50 Hz harmonics" (p. 148).
    Use mains_hz=60 in the Americas.
    """
    step = mains_hz * radar.pri * radar.hdr["range_res_km"]
    return np.arange(1, int(max_km / step) + 1) * step


def object_flag(radar, det, kind, i, j, score):
    """Why a detection is probably NOT an object in the sea ('' if none).

    - 'all-bearings': a real object is seen in one beam (~12 deg) plus weak
      sidelobes. An echo just as strong in most directions at the same range
      does not come from one place: interference, or the radar itself.
    - 'mains-hum': its range matches a power-line harmonic (see above).
    - 'near-site': within 5 km, the coast and harbour around the antennas.
    """
    flags = []
    row = score[i]
    if np.mean(row >= score[i, j] - 6) > 0.5:
        flags.append("all-bearings")
    if kind == "fixed" or abs(det["moving_hz"][i, j]) < 0.1:
        hum = mains_ranges_km(radar, det["rng"][-1] + 5)
        if np.any(np.abs(hum - det["rng"][i]) <= 1.5 * radar.hdr["range_res_km"]):
            flags.append("mains-hum")
    if det["rng"][i] < 5:
        flags.append("near-site")
    return "+".join(flags)


def report_objects(radar, det, path, max_rows=15):
    """Print and save (CSV) the detections, strongest first."""
    rows = []
    for kind, score in (("moving", det["moving_snr"]), ("fixed", det["static_snr"])):
        for i, j in _local_peaks(score, det["min_snr_db"]):
            r, b = det["rng"][i], radar.bearings[j]
            f = det["moving_hz"][i, j] if kind == "moving" else 0.0
            lat, lon = _latlon(radar, r, b)
            rows.append((kind, score[i, j], r, b, f, -f * radar.lam / 2,
                         det["transient_db"][i, j], lat, lon,
                         object_flag(radar, det, kind, i, j, score)))
    rows.sort(key=lambda x: (x[-1] != "", -x[1]))         # candidates first
    det["rows"] = rows
    with open(path, "w") as fh:
        fh.write("kind,snr_db,range_km,bearing_deg,doppler_hz,radial_speed_ms,"
                 "transient_db,lat,lon,flag\n")
        for row in rows:
            fh.write("%s,%.1f,%.2f,%.0f,%.4f,%.2f,%.1f,%.5f,%.5f,%s\n" % row)
    print(f"\nobject detection      : {det['n_windows']} windows of {det['n_fft']} chirps, "
          f"threshold {det['min_snr_db']:.0f} dB above noise, sea zone +/- "
          f"{det['sea_hw']:.3f} Hz around +/- fB")
    ok = [r for r in rows if not r[-1]]
    print(f"detections            : {len(rows)} peaks, {len(ok)} object candidates, "
          f"{len(rows) - len(ok)} flagged as artifacts  (all in {path})")
    if rows:
        print("  kind     SNR  range  bearing  Doppler  radial v  transient  flag")
        for k, snr, r, b, f, v, tr, lat, lon, flag in rows[:max_rows]:
            print(f"  {k:6s} {snr:5.1f} {r:6.1f} {b:7.0f} {f:+8.3f} {v:+8.2f}  "
                  f"{tr:6.1f}     {flag or '-'}")


def plot_objects(radar, rng, det, path):
    """Range x bearing maps of the three scores, plus the range-Doppler map
    at the bearing of the strongest moving detection."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    for a, key, title, vmax in (
            (ax[0], "moving_snr", "Moving objects: strongest non-sea echo", 40),
            (ax[1], "static_snr", "Fixed objects: zero-Doppler echo", 40),
            (ax[2], "transient_db", "Passing objects: mean / median", 10)):
        im = a.pcolormesh(radar.bearings, rng, det[key], shading="auto",
                          cmap="viridis", vmin=0, vmax=vmax)
        a.set(xlabel="bearing [deg true]", ylabel="range [km]", title=title)
        fig.colorbar(im, ax=a, label="dB")
    # Red circles: object candidates. White crosses: flagged as artifacts.
    for kind, snr, r, b, f, v, tr, lat, lon, flag in det["rows"]:
        a = ax[0] if kind == "moving" else ax[1]
        if flag:
            a.plot(b, r, "x", color="w", ms=6, mew=1.2)
        else:
            a.plot(b, r, "o", mfc="none", mec="r", ms=10, mew=1.5)
    for k, a in ((0, ax[0]), (0, ax[1])):
        for hum in mains_ranges_km(radar, rng[-1]):
            a.axhline(hum, color="w", ls=":", lw=0.6, alpha=0.6)

    # Range-Doppler map at the bearing of the strongest moving candidate.
    cands = [r for r in det["rows"] if r[0] == "moving" and not r[-1]]
    if cands:
        j0 = int(np.argmin(np.abs(radar.bearings - cands[0][3])))
        i0 = int(np.argmin(np.abs(rng - cands[0][2])))
    else:
        s = np.nan_to_num(det["moving_snr"], nan=-np.inf)
        i0, j0 = np.unravel_index(np.argmax(s), s.shape)
    im = ax[3].pcolormesh(det["freqs"], rng, det["snr"][:, :, j0].T, shading="auto",
                          cmap="magma", vmin=0, vmax=50)
    for f in (-radar.f_bragg, radar.f_bragg):
        ax[3].axvspan(f - det["sea_hw"], f + det["sea_hw"], color="cyan", alpha=0.12)
    ax[3].plot(det["moving_hz"][i0, j0], rng[i0], "o", mfc="none", mec="cyan", ms=10)
    ax[3].set(xlabel="Doppler [Hz]", ylabel="range [km]",
              title=f"Range-Doppler at {radar.bearings[j0]:.0f} deg (shaded = sea zone)")
    fig.colorbar(im, ax=ax[3], label="dB above noise")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 8. PUT IT TOGETHER AND PLOT
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("sort_file")
    ap.add_argument("site_file", help="JSON with antenna positions (see header)")
    ap.add_argument("--range-km", type=float, default=None,
                    help="range cell to plot (default: 1/3 of max range)")
    ap.add_argument("--bearing", type=float, default=None,
                    help="bearing for the single-spectrum panel (default: boresight)")
    ap.add_argument("--nfft", type=int, default=512,
                    help="chirps per Doppler sub-window (default 512)")
    ap.add_argument("--targets", action="store_true",
                    help="also look for ships and fixed objects (section 9)")
    ap.add_argument("--nfft-targets", type=int, default=128,
                    help="chirps per window for object detection (default 128)")
    ap.add_argument("--min-snr-targets", type=float, default=15.0,
                    help="dB above noise to report an object (default 15)")
    ap.add_argument("--out", default="beamforming_sort",
                    help="output prefix for .png and .npz")
    args = ap.parse_args(argv)

    hdr, data = read_sort(args.sort_file)
    with open(args.site_file) as f:
        site = json.load(f)
    radar = Radar(hdr, site)
    radar.report()

    beamed = beamform(radar, data)                             # (t, r, bearing)
    freqs, power, n_sub = doppler_spectrum(radar, beamed, n_fft=args.nfft)
    v, snr = radial_current(radar, freqs, power)               # (r, bearing)
    print(f"Doppler               : {n_sub} sub-spectra of {min(args.nfft, hdr['n_chirps'])} "
          f"chirps -> resolution {1 / (min(args.nfft, hdr['n_chirps']) * radar.pri):.4f} Hz "
          f"= {radar.lam / 2 / (min(args.nfft, hdr['n_chirps']) * radar.pri) * 100:.1f} cm/s")

    rng = hdr["ranges_km"]
    r_km = args.range_km if args.range_km is not None else rng[-1] / 3
    ir = int(np.argmin(np.abs(rng - r_km)))
    brg = args.bearing if args.bearing is not None else radar.boresight
    jb = int(np.argmin(np.abs(radar.bearings - brg)))
    print(f"\nrange cell {ir} ({rng[ir]:.1f} km), bearing {radar.bearings[jb]:.0f} deg: "
          f"radial current {v[ir, jb]:+.3f} m/s, Bragg SNR {snr[ir, jb]:.1f} dB")
    print(f"valid radial currents : {np.isfinite(v).sum()} of {v.size} cells")

    # Express every spectrum in dB above the noise floor of its own range cell,
    # as the book does for its Toulon spectra (sec. 11.3, p. 147). Otherwise a
    # huge echo near the radar sets the scale and hides everything further out.
    spec_db = 10 * np.log10(power / noise_floor(radar, freqs, power)[None, :, None])
    band = np.abs(freqs) < 3 * radar.f_bragg

    # The full directional Doppler cube, Doppler x range x bearing, kept to
    # +/- 3 f_Bragg (beyond that it is noise) to keep the file small.
    np.savez_compressed(
        f"{args.out}.npz", ranges_km=rng, bearings_deg=radar.bearings,
        radial_current_ms=v, bragg_snr_db=snr, lat=hdr["lat"], lon=hdr["lon"],
        doppler_hz=freqs[band], spectra_db=spec_db[band].astype(np.float32),
        f_bragg_hz=radar.f_bragg)
    plot_cube(radar, rng, freqs[band], spec_db[band], jb, f"{args.out}_cube.png")
    plot_max_doppler(radar, rng, spec_db[band], hdr, f"{args.out}_max_doppler.png")

    if args.targets:
        det = detect_objects(radar, beamed, rng, n_fft=args.nfft_targets,
                             min_snr_db=args.min_snr_targets)
        report_objects(radar, det, f"{args.out}_objects.csv")
        plot_objects(radar, rng, det, f"{args.out}_objects.png")
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))

    # (a) beam pattern of this array --------------------------------------
    look = np.linspace(radar.boresight - 90, radar.boresight + 90, 1801)
    for kind in ("uniform", "hann"):
        ax[0, 0].plot(look, beam_pattern(radar, radar.boresight, look, kind),
                      lw=1.2, label=kind)
    ax[0, 0].set(xlabel="bearing [deg true]", ylabel="response [dB]", ylim=(-50, 2),
                 title=f"Beam pattern of this array, steered at {radar.boresight:.0f} deg")
    ax[0, 0].legend(frameon=False, fontsize=9)
    ax[0, 0].grid(alpha=0.25)

    # (b) bearing / Doppler map at one range -------------------------------
    im = ax[0, 1].pcolormesh(radar.bearings, freqs[band], spec_db[band, ir],
                             shading="auto", vmin=0, vmax=60, cmap="magma")
    for f in (-radar.f_bragg, radar.f_bragg):
        ax[0, 1].axhline(f, color="cyan", lw=0.7, ls="--", alpha=0.7)
    ax[0, 1].set(xlabel="bearing [deg true]", ylabel="Doppler [Hz]",
                 title=f"Directional Doppler spectra at {rng[ir]:.1f} km")
    fig.colorbar(im, ax=ax[0, 1], label="dB above noise")

    # (c) one spectrum ------------------------------------------------------
    ax[1, 0].plot(freqs[band], spec_db[band, ir, jb], lw=1.0)
    for f in (-radar.f_bragg, radar.f_bragg):
        ax[1, 0].axvline(f, color="k", lw=0.7, ls="--", alpha=0.5)
    ax[1, 0].set(xlabel="Doppler [Hz]", ylabel="power [dB above noise]",
                 title=f"Spectrum at {rng[ir]:.1f} km, {radar.bearings[jb]:.0f} deg: "
                       f"v = {v[ir, jb]:+.2f} m/s")
    ax[1, 0].grid(alpha=0.25)

    # (d) radial current map -----------------------------------------------
    lim = np.nanpercentile(np.abs(v), 98) if np.isfinite(v).any() else 1.0
    im = ax[1, 1].pcolormesh(radar.bearings, rng, v, shading="auto",
                             cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1, 1].set(xlabel="bearing [deg true]", ylabel="range [km]",
                 title="Radial current (+ = away from radar)")
    fig.colorbar(im, ax=ax[1, 1], label="m/s")

    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=130)
    extra = f", {args.out}_objects.png, {args.out}_objects.csv" if args.targets else ""
    print(f"\nwrote {args.out}.png, {args.out}_cube.png, {args.out}_max_doppler.png"
          f"{extra} and {args.out}.npz")


def plot_cube(radar, rng, freqs, spec_db, jb, path, n_panels=8):
    """Show the 3-D cube (Doppler x range x bearing) as 2-D slices.

    Left: bearing x Doppler maps at several ranges, near to far.
    Right: range x Doppler map at one bearing, the "range-Doppler map" of
    book sec. 11.4 (Fig. 11.7), here for a single direction.
    """
    picks = np.linspace(0, len(rng) - 1, n_panels).round().astype(int)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, n_panels // 2 + 2)
    for k, ir in enumerate(picks):
        ax = fig.add_subplot(gs[k // (n_panels // 2), k % (n_panels // 2)])
        ax.pcolormesh(radar.bearings, freqs, spec_db[:, ir], shading="auto",
                      vmin=0, vmax=60, cmap="magma")
        for f in (-radar.f_bragg, radar.f_bragg):
            ax.axhline(f, color="cyan", lw=0.5, ls="--", alpha=0.6)
        ax.set_title(f"{rng[ir]:.1f} km", fontsize=10)
        ax.tick_params(labelsize=8)
        if k % (n_panels // 2) == 0:
            ax.set_ylabel("Doppler [Hz]", fontsize=9)
        if k >= n_panels // 2:
            ax.set_xlabel("bearing [deg]", fontsize=9)

    ax = fig.add_subplot(gs[:, n_panels // 2:])
    im = ax.pcolormesh(freqs, rng, spec_db[:, :, jb].T, shading="auto",
                       vmin=0, vmax=60, cmap="magma")
    for f in (-radar.f_bragg, radar.f_bragg):
        ax.axvline(f, color="cyan", lw=0.7, ls="--", alpha=0.6)
    ax.set(xlabel="Doppler [Hz]", ylabel="range [km]",
           title=f"Range-Doppler map at bearing {radar.bearings[jb]:.0f} deg")
    fig.colorbar(im, ax=ax, label="dB above noise")
    fig.suptitle("Directional Doppler cube: bearing x Doppler at 8 ranges (left), "
                 "range x Doppler at one bearing (right)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_max_doppler(radar, rng, spec_db, hdr, path):
    """Range x bearing map coloured by the strongest echo over all Doppler bins.

    The cube is squashed along its Doppler axis by keeping the maximum, so each
    radar cell shows its brightest echo whatever its speed. Mostly that is the
    sea (Bragg peaks); bands at one range across all bearings are 50 Hz mains
    hum (book sec. 11.4, p. 148).
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.pcolormesh(radar.bearings, rng, spec_db.max(axis=0), shading="auto",
                       cmap="viridis", vmin=0, vmax=60)
    fig.colorbar(im, ax=ax, label="max echo over Doppler [dB above noise]")
    ax.set(xlabel="bearing [deg true]", ylabel="range [km]",
           title=f"Strongest echo (any Doppler)\n{hdr['site']}  {hdr['time']}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
