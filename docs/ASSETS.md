# Simulation assets

The repository includes the local Sionna scene and a matching Isaac Sim stage
candidate from the `Eng1Building23Floor5` asset collection. The two simulators
run separately. Follow the repository README for the ROS and Sionna processes.

## Sionna scene

Load [`21202_building_full_room_shifted.xml`](../assets/sionna/21202_building_full_room_shifted.xml).
Its 332 referenced PLY meshes are included in `assets/sionna/meshes/`; all
relative mesh references were checked. The XML and meshes are byte-for-byte
copies of the local source assets. Their combined size is 2,779,729 bytes.
The scene has no external texture dependency.

The reference worker environment was Python 3.12.12, Sionna RT 1.2.1,
Mitsuba 3.7.1, Dr.Jit 1.2.0, NumPy 1.26.4, and einops 0.8.1. These versions
were read from the installed environment. The worker uses `sionna.rt`,
NumPy, and einops; it does not use ROS Python modules.

## Isaac Sim stage

In Isaac Sim 4.1, open
[`21_bldg_environment_and_robot_shifted.usd`](../assets/isaac/21_bldg_environment_and_robot_shifted.usd).
Keep the entire `assets/isaac/` directory together. Its building, robot,
sensor, material, and texture files use relative references in the copied
stage. There are 85 Isaac files, approximately 206 MiB in total. The stage's
robot layer includes ROS1 graphs for simulation time,
odometry, ground-truth transforms, IMU, laser scan, and velocity commands.
Enable the Isaac Sim ROS1 bridge and configure the ROS master as described
in the repository README before starting the timeline.

This stage matches the expected robot and ROS1 interfaces. It has **not**
been verified as the exact stage used for example run 8985, and no fresh
Isaac simulation was performed while preparing this repository. Do not
interpret the bundled example animation as validation of a new run.

Only the published copies were changed:

- Absolute building and robot payload paths were converted to relative paths.
- The active local-Nucleus RealSense reference was replaced with the bundled
  `robot/RealSense/rsd455.usd` copy.
- Obsolete deleted reference/payload opinions to historical assets, including
  the conflicting deletion of the now-active local sensor, were removed.
- A personal converter-source path in the Scout layer's documentation was
  replaced with its logical source asset path.

Original assets were not modified. Local composition dependencies, USD
asset attributes, relative MDL imports, and MDL texture defaults were checked
using the installed Isaac Sim 4.1 USD library. The copied USD layers contain
no personal home-directory paths. Every local asset dependency resolves
inside the copied tree. No individual bundled asset reaches 100 MiB.

### External material resources

The RealSense camera retains these three NVIDIA material references:

- [Aluminum_Anodized.mdl](http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Metals/Aluminum_Anodized.mdl)
- [Aluminum_Cast.mdl](http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Metals/Aluminum_Cast.mdl)
- [Plastic_ABS.mdl](http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Plastics/Plastic_ABS.mdl)

These upstream resources were not downloaded or verified. Isaac may need
network access or its existing asset cache to resolve them and any resources
they reference. `OmniPBR.mdl` and `OmniGlass.mdl` come from Isaac's material
module search path. The asset bundle is therefore not a completely offline
Isaac distribution.

## Provenance and existing notices

[`assets/manifest.json`](../assets/manifest.json) records each original
logical relative path, original SHA256, published SHA256, byte counts, and
any changes. It intentionally omits personal absolute source paths.

Existing copyright and redistribution notices are preserved in
`Materials/OmniUe4Base.mdl`, `Materials/OmniUe4Function.mdl`, and
`Materials/OmniUe4Translucent.mdl`. Those files contain NVIDIA 2020 notices.
The source scene XML and PLY mesh headers contain no explicit license
statement. This document does not assign a new license to simulation assets
or extend the ROS package's license declaration to them.
