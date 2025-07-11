-- Set up mmWaveStudio.exe, copy from Startup.lua
local debugging = false
local rttt_package = "V15"

RTTT = RSTD
function Load_AR1x_Client(automation_mode)
	local al_path
	local client_path
	local client_gui_path
	local registers_xml
	local ts18_controller_path
	local ocla_cc3100_controller_path
	local acmb_controller_path
	local bs_gui_controller_path
	
	AR1_GUI = true
	
	-- Set paths for AL and client DLLs
	ar1x_controller_path = RSTD_PATH .. "\\Clients\\AR1xController\\AR1xController.dll"
    registers_xml = RSTD_PATH .. "\\Clients\\AR1xController\\AR2944ES1P0_Registers.xml"
	al_path = RSTD_PATH .. "\\RunTime\\SAL.dll"	
	client_path = RSTD_PATH .. "\\Clients\\\\LabClient.dll"
	client_gui_path = ""
		
	-- Unbuild a previous build
	RSTD.UnBuild()
	
	-- Set the AL and Clients to build
	RSTD.SetClientDll(al_path, client_path, client_gui_path, 0)

	-- Get updated variables values automatically for the wlan GUI
	RSTD.SetVar ("/Settings/AutoUpdate/Enabled" , "TRUE")
	RSTD.SetVar ("/Settings/AutoUpdate/Interval" , "1")
	
	-- Set if to update monitored variables display in the BrowseTree
	RSTD.SetVar ("/Settings/Monitors/UpdateDisplay" , "TRUE")

	-- Set to have a click on MonitorStart/MonitorStop in GUI automatically call GO/Stop
	RSTD.SetVar ("/Settings/Monitors/OneClickStart" , "TRUE")
	RSTD.SetVar ("/Settings/Automation/Automation Mode" , tostring(automation_mode))
	RSTD.Transmit("/")
	RSTD.SaveClientSettings()
	
	-- Run the TestScripter mapping functions
	dofile(RSTD_PATH .. "\\Scripts\\Startup\\General_functions.lua")
    dofile(RSTD_PATH .. "\\Scripts\\Startup\\BinDecHex.lua")
    dofile(RSTD_PATH .. "\\Scripts\\Startup\\lib_math.lua")

	RSTD.Build()
	RSTD.RegisterDllEx(ar1x_controller_path, false)
	RSTD.SetExternalAL(ar1x_controller_path)		
	RSTD.SetTitle(ar1.GuiVersion())	
	RSTD.LoadExpose(registers_xml)
	ar1.ShowGui()		
end

-- Get the file parameters
DUT_VERSION = ...
-- RT3 installation path
RSTD_PATH = RSTD.GetRstdPath()
-- Options for DUT Version
DUT_VER = {AR1xxx = 1}
-- Set the target device
DUT_VERSION = DUT_VERSION or DUT_VER.AR1xxx
-- Set automoation mode on/off (no message boxes)
local automation_mode = false

-- Display timestmaps in output/log
RSTD.SetAndTransmit ("/Settings/Scripter/Display DateTime" , "1")
RSTD.SetAndTransmit ("/Settings/Scripter/DateTime Format" , "HH:mm:ss")

Load_AR1x_Client(automation_mode)
TESTING = false
WriteToLog("TESTING = ".. tostring(TESTING) .. "\n", "green")

--RSTD.NetClose()
RSTD.NetStart()

-- Run configuration and data capturing
ar1.FullReset()
ar1.SOPControl(2)

ar1.Connect(8, 115200, 1000)

ar1.Calling_IsConnected()
ar1.deviceVariantSelection("XWR2944")
ar1.frequencyBandSelection("77G")
ar1.SelectChipVersion("AWR2944")
ar1.DownloadBSSFw("C:\\ti\\mmwave_studio_03_01_04_04\\rf_eval_firmware\\radarss\\xwr29xx_radarss_rprc.bin")
ar1.GetBSSFwVersion()
ar1.GetBSSPatchFwVersion()
ar1.DownloadMSSFw("C:\\ti\\mmwave_studio_03_01_04_04\\rf_eval_firmware\\masterss\\awr2xxx_mmwave_full_mss_rprc.bin")
ar1.GetMSSFwVersion()
ar1.PowerOn(0, 1000, 0, 0)
ar1.RfEnable()
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0)
ar1.LPModConfig(0, 0)
ar1.RfInit()
-- ar1.SetCalMonFreqLimitConfig(77,81)
ar1.DataPathConfig(513, 1216644097, 0)
ar1.LvdsClkConfig(1, 1)
ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)
-- ar1.ProfileConfig(UInt16 profileId, Double startFreqConst, Single idleTimeConst, Single adcStartTimeConst, 
-- Single rampEndTime, UInt32 tx0OutPowerBackoffCode, UInt32 tx1OutPowerBackoffCode, UInt32 tx2OutPowerBackoffCode, UInt32 tx3OutPowerBackoffCode, Single tx0PhaseShifter, Single tx1PhaseShifter, Single tx2PhaseShifter, Single tx3PhaseShifter, Single freqSlopeConst, Single txStartTime, UInt16 numAdcSamples, UInt16 digOutSampleRate, UInt32 hpfCornerFreq1, 
-- UInt32 hpfCornerFreq2, Char rxGain, Char hpfInitControlSelect, Char highResTxPowerEn, Char runTimeTxPowMultiTxCalEn) - Profile configuration API which defines chirp profile parameters
ar1.ProfileConfig(0, 77, 10, 6, 120, 0, 0, 0, 0, 0, 0, 0, 0, 29.982, 0, 1024, 10000, 2216755200, 0, 30, 0, 0, 0)
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0)
ar1.ChirpConfig(2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0)
ar1.ChirpConfig(3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1)
-- ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 0, 0, 1)
ar1.DisableTestSource(0)
ar1.FrameConfig(0, 3, 256, 255, 240, 1, 66.67)
ar1.GetCaptureCardDllVersion()
ar1.SelectCaptureDevice("DCA1000")
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
ar1.CaptureCardConfig_PacketDelay(25)
ar1.GetCaptureCardFPGAVersion()

ar1.CaptureCardConfig_StartRecord("D:\\rawData\\cap7\\data.bin", 1)
ar1.StartFrame()

-- Exit mmWaveStudio.exe
-- os.exit()