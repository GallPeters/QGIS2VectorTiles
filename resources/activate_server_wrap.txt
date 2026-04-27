Set WshShell = CreateObject("WScript.Shell")
base = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
WshShell.Run "cmd /c """ & base & "utils\activate_server.bat""", 0
Set WshShell = Nothing