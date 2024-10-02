# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- SourceSimulator class that allows adding point sources with custom locations and waveforms to the simulation ([3bad4a8](https://github.com/ctrltz/meegsim/commit/3bad4a86a3712beb43fb404481c15e1a54250d87), [#24](https://github.com/ctrltz/meegsim/pull/24))
- Implementation of patch sources ([#28](https://github.com/ctrltz/meegsim/pull/28))
- Waveforms of white noise, narrowband oscillation (white noise filtered in a narrow frequency band) and 1/f noise with adjustable slope ([#8](https://github.com/ctrltz/meegsim/pull/8))
- Random vertices in the whole source space or in a subset of vertices as location for point sources ([#10](https://github.com/ctrltz/meegsim/pull/10))
- Adjustment of the SNR of the point sources based on sensor space power ([#9](https://github.com/ctrltz/meegsim/pull/9), [#31](https://github.com/ctrltz/meegsim/pull/31))
- Phase-phase coupling with a constant phase lag or a probabilistic phase lag according to the von Mises distribution ([#11](https://github.com/ctrltz/meegsim/pull/11))
- Traversal of the coupling graph to ensure that the coupling is set up correctly when multiple connectivity edges are defined ([#12](https://github.com/ctrltz/meegsim/pull/12))
