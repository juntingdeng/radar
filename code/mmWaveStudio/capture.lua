-- Set up mmWaveStudio.exe, copy from Startup.lua

-- ar1.GetCaptureCardDllVersion()
-- ar1.SelectCaptureDevice("DCA1000")
-- ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
-- ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
-- ar1.CaptureCardConfig_PacketDelay(25)
-- ar1.GetCaptureCardFPGAVersion()

ar1.CaptureCardConfig_StartRecord("D:\\rawData\\cap10\\data.bin", 1)
ar1.StartFrame()

-- Exit mmWaveStudio.exe
-- os.exit()