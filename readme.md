Setup: Refer to /docs/ti_awr2944_custom_user_guide for hardware connection.

Software: 
    Uniflash: Flash radar sensor. Flashing is not necessary unless required.
    mmWaveStudio: Configure and control radar sensor, Windows computer required. Please don’t use a school computer, as the firewall needs to be disabled for data transmission over Ethernet.

Hardware:
    AWE2944EVM:
        12 V power supply
        2 USB-MicroUSB cables

    DCA100EVM:
        5 V power supply
        1 Ethernet cable

Data collection:
    1. Initialize radar configuration by running '/radar/init.py'. Change radar configuration by modifying server.lua.
    2. Run' /radar/collect.py' to store raw radar data to .h5 file.
    3. Run 'process.py --cfg_file --data_file' to get processed range-angle spectrum.
    4. Run 'process.py --cfg_file --live' to realize data processing while collecting data.