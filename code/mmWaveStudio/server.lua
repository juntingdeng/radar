--    ______  _____  _    _ _______  ______
--   |_____/ |     |  \  /  |______ |_____/
--   |    \_ |_____|   \/   |______ |    \_
--   Radar Control Server for mmWave Studio
--

-- Initialize mmWave Studio ---------------------------------------------------

-- Taken from TI-supplied code. These lines call the default initialization
-- script which usually runs when mmWave studio starts.

-- mmwavestudio installation path
RSTD_PATH = RSTD.GetRstdPath()
-- Declare the loading function
dofile(RSTD_PATH .. "\\Scripts\\Startup.lua")

-- Initialize Radar -----------------------------------------------------------

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
ar1.LPModConfig(0, 1)
ar1.RfInit()
-- ar1.SetCalMonFreqLimitConfig(77,81)
ar1.DataPathConfig(513, 1216644097, 0)
ar1.LvdsClkConfig(1, 1)
ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)
-- ar1.ProfileConfig(UInt16 profileId, Double startFreqConst, Single idleTimeConst, Single adcStartTimeConst, 
-- Single rampEndTime, UInt32 tx0OutPowerBackoffCode, UInt32 tx1OutPowerBackoffCode, UInt32 tx2OutPowerBackoffCode, UInt32 tx3OutPowerBackoffCode, Single tx0PhaseShifter, Single tx1PhaseShifter, Single tx2PhaseShifter, Single tx3PhaseShifter, Single freqSlopeConst, Single txStartTime, UInt16 numAdcSamples, UInt16 digOutSampleRate, UInt32 hpfCornerFreq1, 
-- UInt32 hpfCornerFreq2, Char rxGain, Char hpfInitControlSelect, Char highResTxPowerEn, Char runTimeTxPowMultiTxCalEn) - Profile configuration API which defines chirp profile parameters
ar1.ProfileConfig(0, 77, 10, 6, 120, 0, 0, 0, 0, 0, 0, 0, 0, 29.982, 0, 1024, 10000, 2216755200, 0, 30, 0, 0, 0) -- todo adjust adc sample rate
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0)
ar1.ChirpConfig(2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0)
ar1.ChirpConfig(3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1)
-- ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 0, 0, 1)
ar1.DisableTestSource(0)
ar1.FrameConfig(0, 3, 0, 64, 240, 1, 66.67)
ar1.GetCaptureCardDllVersion()
ar1.SelectCaptureDevice("DCA1000")
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
ar1.CaptureCardConfig_PacketDelay(25)
ar1.GetCaptureCardFPGAVersion()

print("Initialization complete.")

-- Server ---------------------------------------------------------------------
-- WORK_DIR = "G:\\My Drive\\CMU\\Research\\3DImage\\sensor\\TI\\setup_test"
function read()
    local file = io.open("G:\\My Drive\\CMU\\Research\\3DImage\\sensor\\TI\\setup_test\\code\\radar\\test\\msg", "r")
    if not file then
        return nil
    end
    local msg = file:read("*a")
    print("read msg: ", msg)
    file:close()
    os.remove("G:\\My Drive\\CMU\\Research\\3DImage\\sensor\\TI\\setup_test\\code\\radar\\test\\msg")
    return msg
end

running = false
while true do
    msg = read()
    if msg == "start" then
        if running == false then
            print("Starting capture...")
            ar1.CaptureCardConfig_StartRecord("G:\\My Drive\\CMU\\Research\\3DImage\\sensor\\TI\\setup_test\\code\\radar\\test\\tmp.bin", 1)
            ar1.StartFrame()
            running = true
        else
            print("Tried to start an already-running radar.")
        end
    elseif msg == "stop" then
        if running == true then
            print("Stopping capture...")
            ar1.StopFrame(0)
            print("Stopping capture Done")
            running = false
           
        else
            print("Tried to stop an already-stopped radar.")
        end
    elseif msg == "exit" then
        print("Exiting...")
        os.exit()
    end
    RSTD.Sleep(1000)
end
