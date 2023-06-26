#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else 
#include <CL/cl.h>
#endif

#include <cassert>
#include <iostream>

int main(){
    std::cout << "starting device query" << std::endl;

    cl_uint l_err = CL_SUCCESS;

    // number of platforms
    cl_uint l_n_platforms = 0;
    l_err = clGetPlatformIDs( 0, 
                              NULL, 
                              &l_n_platforms );
    assert( l_err == CL_SUCCESS );
    std::cout << "number of platforms: " << l_n_platforms << std::endl;

    // platform IDs
    cl_platform_id *l_platform_ids = new cl_platform_id[ l_n_platforms ];
    l_err = clGetPlatformIDs( l_n_platforms, 
                              l_platform_ids, 
                              NULL);
    assert( l_err == CL_SUCCESS );

    for (cl_uint l_pl =0; l_pl < l_n_platforms; l_pl++)
    {
        cl_platform_id l_pid = l_platform_ids[l_pl];

        cl_char l_tmp_string[8192] = {0};

        std::cout << "gathering info for platform with id: " << l_pid << std::endl;

        // platform profile
        l_err = clGetPlatformInfo(  l_pid, 
                                    CL_PLATFORM_PROFILE, 
                                    sizeof(l_tmp_string), 
                                    &l_tmp_string, 
                                    NULL );   

        assert( l_err == CL_SUCCESS);

        std::cout << "  CL_PLATFORM_PROFILE: " << l_tmp_string << std::endl;

        // platform version
        l_err = clGetPlatformInfo(  l_pid, 
                                    CL_PLATFORM_VERSION, 
                                    sizeof(l_tmp_string), 
                                    &l_tmp_string, 
                                    NULL );   

        assert( l_err == CL_SUCCESS);

        std::cout << "  CL_PLATFORM_VERSION: " << l_tmp_string << std::endl;

        // platform name
        l_err = clGetPlatformInfo(  l_pid, 
                                    CL_PLATFORM_NAME, 
                                    sizeof(l_tmp_string), 
                                    &l_tmp_string, 
                                    NULL );   

        assert( l_err == CL_SUCCESS);

        std::cout << "  CL_PLATFORM_NAME: " << l_tmp_string << std::endl;

        // number of devices
        cl_uint l_n_devices = 0;
        l_err = clGetDeviceIDs( l_pid,
                                CL_DEVICE_TYPE_ALL,
                                0, 
                                NULL, 
                                &l_n_devices );
        assert( l_err == CL_SUCCESS );
        std::cout << "  number of devices: " << l_n_platforms << std::endl;


        cl_device_id *l_device_ids = new cl_device_id[l_n_devices];
        l_err = clGetDeviceIDs( l_pid,
                                CL_DEVICE_TYPE_ALL,
                                l_n_devices,
                                l_device_ids, 
                                NULL );
        assert( l_err == CL_SUCCESS );
        
        for( cl_uint l_de = 0; l_de < l_n_devices; l_de++ ){
            cl_device_id l_did = l_device_ids[l_de];

            std::cout << "  gathering information for devide with id: " << l_did << std::endl;


            l_err = clGetDeviceInfo(    l_did,
                                        CL_DEVICE_NAME,
                                        sizeof(l_tmp_string),
                                        &l_tmp_string, 
                                        NULL );
            assert( l_err == CL_SUCCESS );

            std::cout << "  CL_DEVICE_NAME: " << l_tmp_string << std::endl;

            l_err = clGetDeviceInfo(    l_did,
                                        CL_DEVICE_OPENCL_C_VERSION,
                                        sizeof(l_tmp_string),
                                        &l_tmp_string, 
                                        NULL );
            assert( l_err == CL_SUCCESS );

            std::cout << "  CL_DEVICE_OPENCL_C_VERSION: " << l_tmp_string << std::endl;

            cl_ulong l_global_mem_size = 0;
            l_err = clGetDeviceInfo(    l_did,
                                        CL_DEVICE_GLOBAL_MEM_SIZE,
                                        sizeof(l_global_mem_size),
                                        &l_global_mem_size, 
                                        NULL );
            assert( l_err == CL_SUCCESS );

            std::cout << "  CL_DEVICE_GLOBAL_MEM_SIZE: " << l_global_mem_size << std::endl;
        }
    }
    

    delete [] l_platform_ids;

    std::cout << "device query ended" << std::endl;
}