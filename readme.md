Setup: 

Refer to /docs/ti_awr2944_custom_user_guide for hardware connection.

Software: 

    Uniflash: Flash radar sensor. Flashing is not necessary unless required.
    mmWaveStudio: Configure and control radar sensor, Windows computer required. Please don’t use a school computer, as the firewall needs to be disabled for data transmission over Ethernet.

Hardware:

    AWE2944EVM:
        12 V power supply
        2 USB-MicroUSB cables
    DCA1000:
        5 V power supply
        1 Ethernet cable

Radar Data collection and processing:

    1. Initialize radar configuration by running '/radar/init.py'. Change radar configuration by modifying /mmWaveStudio/server.lua.
    2. Run' /radar/collect.py' to store raw radar data to .h5 file.
    3. Run 'process.py --cfg_file --data_file' to get processed range-angle spectrum.
    4. Run 'process.py --cfg_file --live' to realize data processing while collecting data.

RGB and Radar data collection:

    1. Run '/mmWaveStudio/socket_server.py' to initialize radar configuration and wait for the 'Start' message from camera to start data collection.
    2. Run '/mmWaveStudio/socket_client.py' to start RGB video collection and send 'Start' message.
    3. Stop camera after finishing radar data collection.
