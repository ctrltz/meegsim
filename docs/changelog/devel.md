# [Unreleased]

### Added

- A desired level of white noise can be added in sensor space to model measurement
noise ([#58](https://github.com/ctrltz/meegsim/pull/58))
- A possibility to plot the source configuration ([#59](https://github.com/ctrltz/meegsim/pull/59))
- Adjustment of global (all signal vs. all noise sources) SNR ([#64](https://github.com/ctrltz/meegsim/pull/64))
- Adjustment of the standard deviation of source activity ([#66](https://github.com/ctrltz/meegsim/pull/66))
- Allow specifying standard deviation via a SourceEstimate object ([#67](https://github.com/ctrltz/meegsim/pull/67))
- A method for setting phase-phase coupling by adding noise to the shifted copy of input waveform ([#71](https://github.com/ctrltz/meegsim/pull/71))
- Function to convert the sources to mne.Label ([#73](https://github.com/ctrltz/meegsim/pull/73))
- Quick dict-like access to the simulated sources ([#82](https://github.com/ctrltz/meegsim/pull/82))
- Partial control over the amplitude envelope of the coupled waveform: same as input or randomly generated ([#87](https://github.com/ctrltz/meegsim/pull/87))

### Changed

- Reworked normalization of source activity: by default, all source time courses are scaled to make their standard deviation equal to 1 nAm ([#66](https://github.com/ctrltz/meegsim/pull/66))
- Improved performance when adjusting the SNR for a large number of sources ([#68](https://github.com/ctrltz/meegsim/pull/68))

### Fixed

- Fixed a bug that caused different sources to have the same location and/or waveform when random state was explicitly provided ([#76](https://github.com/ctrltz/meegsim/pull/76))
