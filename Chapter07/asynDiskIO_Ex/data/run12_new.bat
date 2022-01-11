@echo off
rem
rem the run12_new.bat file was used to obtain the results in table.
rem
rem This bat file runs asyncDisk_Ex.exe 12 times with different input and output files.
rem The wall clock time is displayed between running each job and difference between
rem 2 consecutive times will be a good measure of actual job time including cached writes.
rem
rem The 1st parameter is the path to the exe file, use . if not required.
rem The 2nd parameter is the path to the input data files, use . if not required.
rem Note output files are written to the current directory.
rem
rem Paramters 3-6 set the program parameters frames, ktime, gpu and flush.
rem The other program parameters (blocks, threads and dsize) are hard wired as
rem 256 256 and 2^28 but obviously could be changed by editing this file.  
rem
rem The four rows of table 7.3 show results for four different values of the ktimes 
rem parameter namely 10, 6400, 12800 and 19200.
rem case A read and write to same hard drive
rem case B read ssd write hard drive 
rem
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A1.bin B1.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A2.bin B2.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A3.bin B3.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A4.bin B4.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A5.bin B5.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A6.bin B6.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A7.bin B7.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A8.bin B8.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A9.bin B9.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A10.bin B10.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A11.bin B11.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%
%1\asyncDiskIO_Ex.exe %2\A12.bin B12.bin 256 256 28 %3 %4 %5 %6
@echo %TIME%