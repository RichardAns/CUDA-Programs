The data files in this folder are MRI head scans with 256 x 256 x 256 voxels. 

They contain 16-bit integers in the range 0-32767 and can be processed as either short or ushort types.
The register.cu code in this project is written for the ushort data type.

head.raw is the orignal image

vol1.raw and vol2.raw are versions of head.raw transfromed as discussed in section 5.9.

vol1to2.raw and vol2to1.raw are the results of registering vol1 to vol2 and vice vesa.

Please note these files are too big to store on GitHub thy can be downloads by following this link:

    https://1drv.ms/u/s!AnPXxzdjl4BZhv0vBMQoulILKqlOnw?e=GkgDw9

Richard Ansorge January 2022