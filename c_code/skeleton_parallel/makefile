# Compile executable
skeleton: skeleton.c
	#gcc -Wall skeleton.c simple_tiffio.c -ltiff -o skeleton
	gcc -Wall -fopenmp skeleton.c simple_tiffio.c -Iinclude -ltiff -Llib -o skeleton
	
# Compile library
libskeleton: skeleton.c
	#gcc -Wall -fpic -shared skeleton.c simple_tiffio.c -ltiff -o libskeleton.so
	gcc -Wall -fpic -shared -fopenmp skeleton.c simple_tiffio.c -Iinclude -ltiff -Llib -o libskeleton.so

libskeleton_single: skeleton.c
	gcc -Wall -fpic -shared skeleton.c simple_tiffio.c -Iinclude -ltiff -Llib -o libskeleton_single.so