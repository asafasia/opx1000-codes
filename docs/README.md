# Documentation

Repository documentation that is not tied to one calibration experiment lives
here.

- [Hardware documentation](hardware/README.md): wiring constraints, device
  reference data, and vendor-provided measurements.
- [Calibration routine hierarchy](calibrations/README.md): general bring-up
  order, fine-tuning flow, and practical calibration notes.
- [Device profiles](../profiles/README.md): executable configuration used to
  create the QuAM machine, including `CreateMachine`, profile load/save
  behavior, and Profile Studio.

Hardware documentation records durable facts about the setup. The files under
`profiles/` remain the source of truth for the configuration used by
experiments.
