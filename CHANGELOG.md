# Changelog

All notable changes to this project will be documented in this file. The public
API is defined in the [API reference](https://meegsim.readthedocs.io/en/stable/api/).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 0.0.2 (Unreleased)

### Added

- A desired level of white noise can be added in sensor space to model measurement
noise ([#58](https://github.com/ctrltz/meegsim/pull/58))
- A possibility to plot the source configuration ([#59](https://github.com/ctrltz/meegsim/pull/59))
- Adjustment of global (all signal vs. all noise sources) SNR ([#64](https://github.com/ctrltz/meegsim/pull/64))
- Adjustment of the standard deviation of source activity ([#66](https://github.com/ctrltz/meegsim/pull/66))
- Allow specifying standard deviation via a SourceEstimate object ([#67](https://github.com/ctrltz/meegsim/pull/67))
- A method for setting phase-phase coupling by adding noise to the shifted copy of input waveform ([#71](https://github.com/ctrltz/meegsim/pull/71))
- Function to convert the sources to mne.Label ([#73](https://github.com/ctrltz/meegsim/pull/73))

### Changed

- Reworked normalization of source activity: by default, all source time courses are scaled to make their standard deviation equal to 1 nAm ([#66](https://github.com/ctrltz/meegsim/pull/66))
- Improved performance when adjusting the SNR for a large number of sources ([#68](https://github.com/ctrltz/meegsim/pull/68))

### Fixed

- Fixed a bug that causes different sources to have the same location and/or waveform when random state was explicitly provided ([#76](https://github.com/ctrltz/meegsim/pull/76))

## Version 0.0.1 (2024-10-31)

### Added

- SourceSimulator class that allows adding point and patch sources with custom locations and waveforms to the simulation
- Template waveforms of white noise, narrowband oscillation (white noise filtered in a narrow frequency band) and 1/f noise with adjustable slope
- Random vertices in the whole source space or in a subset of vertices as location for point sources
- Adjustment of the SNR of the sources based on sensor space power
- Phase-phase coupling with a constant phase lag or a probabilistic phase lag according to the von Mises distribution
- Traversal of the coupling graph to ensure that the coupling is set up correctly when multiple connectivity edges are defined
