proj = pa2
srcs = main.cpp include/*
builddir = build

makeflags = -j$(shell expr $(shell nproc) - 1) # Parallel compiling

release: ${srcs} CMakeLists.txt
	cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
		-DCMAKE_BUILD_TYPE="Release"
	@make -C ${builddir} ${makeflags}
debug: ${srcs} CMakeLists.txt
	cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
		-DCMAKE_BUILD_TYPE="Debug"
	@make -C ${builddir} ${makeflags}
clean:
	@make -C ${builddir} clean
purge:
	-rm -rf ${builddir}
fresh: purge all

.PHONY: debug release clean fresh purge

# vim: set ft=make:

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Dec 24 2020, 09:24 [CST]
