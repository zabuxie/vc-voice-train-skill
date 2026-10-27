' Launch the IndexTTS adapter fully detached from the calling console.
' wscript.exe runs it hidden, so the python process is not a child of the
' .cmd wrapper and survives its exit.
'
' ENV_ROOT resolution:
'   1. INDEX_TTS_ENV_ROOT env var, if set
'   2. parent of this script   (script sits in indextts/)
'   3. parent of that, as fallback
Option Explicit

Dim fso, sh, scriptDir, envRoot, py, adapterPath, env

Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

envRoot = sh.ExpandEnvironmentStrings("%INDEX_TTS_ENV_ROOT%")
If envRoot = "%INDEX_TTS_ENV_ROOT%" Or envRoot = "" Then
  envRoot = fso.GetParentFolderName(scriptDir)
  If Not fso.FileExists(envRoot & "\index-tts\.venv\Scripts\python.exe") Then
    envRoot = fso.GetParentFolderName(envRoot)
  End If
End If

py = envRoot & "\index-tts\.venv\Scripts\python.exe"
adapterPath = scriptDir & "\adapter.py"

If Not fso.FileExists(py) Then
  WScript.Echo "python not found: " & py & vbCrLf & "Set INDEX_TTS_ENV_ROOT."
  WScript.Quit 1
End If
If Not fso.FileExists(adapterPath) Then
  WScript.Echo "adapter.py not found next to this script."
  WScript.Quit 1
End If

Set env = sh.Environment("PROCESS")
env("PYTORCH_CUDA_ALLOC_CONF") = "expandable_segments:True"
env("PYTHONIOENCODING") = "utf-8:replace"
env("INDEX_TTS_ENV_ROOT") = envRoot

sh.CurrentDirectory = envRoot

' 0 = hidden window, False = do not wait
sh.Run """" & py & """ """ & adapterPath & """", 0, False
