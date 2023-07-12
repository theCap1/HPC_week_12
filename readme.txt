Useful Commands:

specify device      export ANDROID_SERIAL=3000a4df                                                          // sets global variable
compile             make                                                                                    //  don't forget cross compiler
push to device      adb push build/device_query /data/local/tmp/sven
execute on device   adb shell "LD_LIBRARY_PATH=/data/local/tmp/sven ./data/local/tmp/sven/device_query      // important to have libraries in the specified directorY!!!