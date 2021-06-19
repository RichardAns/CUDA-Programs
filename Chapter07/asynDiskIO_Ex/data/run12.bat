@echo off
rem
rem the run12.bat file was used to obtain the results in table.
rem
rem This bat file runs asyncDisk_Ex.exe 12 times with different input and output files.
rem The wall clock time is displayed between running each job and difference between
rem 2 consecutive times will be a good measure of actual job time including cached writes.
rem
rem The blocks, threads and dsize parameters are hard wired as 256 256 and 2^28 but obviously
rem can be changed by editing the file. The parameters frames, ktime, gpu and flush can be set
rem when invoking this bat file using the first 4 optional parameter. 
rem
rem The 5th parameter can be used to specify a path to the executable program.
rem
rem
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A1.bin B1.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A2.bin B2.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A3.bin B3.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A4.bin B4.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A5.bin B5.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A6.bin B6.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A7.bin B7.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A8.bin B8.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A9.bin B9.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A10.bin B10.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A11.bin B11.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%
%5\asyncDiskIO_Ex.exe C:\temp\A12.bin B12.bin 256 256 28 %1 %2 %3 %4
@echo %TIME%