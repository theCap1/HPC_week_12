#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else 
#include <CL/cl.h>
#endif

#include <cassert>
#include <iostream>

static const char * l_my_triad = R"(
    __kernel void triad(    __global float * i_a,
                            __global float * i_b,
                            __global float * o_c ){
        size_t l_gwid = get_global_id(0);
        o_c[l_gwid] = i_a[l_gwid] + 2.0f * i_b[l_gwid];
    }
)";

int main(){
    std::cout << "starting device query" << std::endl;

    cl_int l_err = CL_SUCCESS;

    // number of platforms
    cl_uint l_n_platforms = 0;
    l_err = clGetPlatformIDs( 0, 
                              NULL, 
                              &l_n_platforms );
    assert( l_err == CL_SUCCESS );
    std::cout << "number of platforms: " << l_n_platforms << std::endl;
    assert( l_n_platforms > 0);

    // platform IDs
    cl_platform_id *l_platform_ids = new cl_platform_id[ l_n_platforms ];
    l_err = clGetPlatformIDs( l_n_platforms, 
                              l_platform_ids, 
                              NULL);
    assert( l_err == CL_SUCCESS );

    cl_char l_tmp_string[8192] = {0};

    // platform name
    l_err = clGetPlatformInfo(  l_platform_ids[0], 
                                CL_PLATFORM_NAME, 
                                sizeof(l_tmp_string), 
                                &l_tmp_string, 
                                NULL );   

    assert( l_err == CL_SUCCESS);

    std::cout << "  CL_PLATFORM_NAME: " << l_tmp_string << std::endl;

    // number of devices
    cl_uint l_n_devices = 0;
    l_err = clGetDeviceIDs( l_platform_ids[0],
                            CL_DEVICE_TYPE_ALL,
                            0, 
                            NULL, 
                            &l_n_devices );
    assert( l_err == CL_SUCCESS );

    cl_device_id *l_device_ids = new cl_device_id[l_n_devices];
    l_err = clGetDeviceIDs( l_platform_ids[0],
                            CL_DEVICE_TYPE_ALL,
                            l_n_devices,
                            l_device_ids, 
                            NULL );
    assert( l_err == CL_SUCCESS );

    l_err = clGetDeviceInfo(    l_device_ids[0],
                                CL_DEVICE_OPENCL_C_VERSION,
                                sizeof(l_tmp_string),
                                &l_tmp_string, 
                                NULL );
    assert( l_err == CL_SUCCESS );

    std::cout << "  CL_DEVICE_OPENCL_C_VERSION: " << l_tmp_string << std::endl;

    /*
     * prepare program execution
     */
    cl_context l_context = clCreateContext(   NULL,
                                    1,
                                    l_device_ids+0,
                                    NULL,
                                    NULL,
                                    &l_err );
    assert( l_err == CL_SUCCESS );

    cl_program l_program = clCreateProgramWithSource(   l_context,
                                                        1,
                                                        &l_my_triad,
                                                        NULL,
                                                        &l_err );
    assert( l_err == CL_SUCCESS );

    /*
     * build program 
     */
    std::cout << "build program: " << std::endl;
    std::cout << l_my_triad << std::endl;
    l_err = clBuildProgram( l_program,
                            0,
                            NULL,
                            NULL,
                            NULL,
                            NULL);

    if ( l_err != CL_SUCCESS )
    {
        std::cerr << "failed to build program!" << std::endl;

        clGetProgramBuildInfo(  l_program,
                                l_device_ids[0],
                                CL_PROGRAM_BUILD_LOG,
                                sizeof(l_tmp_string),
                                l_tmp_string,
                                NULL );
        std::cerr << l_tmp_string << std::endl;
        return 1;
    }else{
        std::cout << "successfully build program" << std::endl;
    }
    
    // create kernel
    cl_kernel l_triad = clCreateKernel(  l_program,
                                        "triad",
                                        &l_err );
    assert( l_err == CL_SUCCESS ); 

    // allocate  host memory
    std::cout << "allocating host memory" << std::endl;
    std::size_t l_n_values = 7;
    float *l_a_host = new float[l_n_values];
    float *l_b_host = new float[l_n_values];
    float *l_c_host = new float[l_n_values];

    // initialize host memory
    std::cout << "initializing host memory" << std::endl;
    for (std::size_t l_en = 0; l_en < l_n_values; l_en++)
    {
        l_a_host[l_en] = l_en;
        l_b_host[l_en] = 3*l_en;
        l_c_host[l_en] = -1;
    }

    std::cout << "allocation device memory" << std::endl;

    cl_mem l_a_device = clCreateBuffer( l_context,
                                        CL_MEM_READ_ONLY, 
                                        sizeof(float)*l_n_values, 
                                        NULL, 
                                        &l_err );
    assert( l_err == CL_SUCCESS ); 

    cl_mem l_b_device = clCreateBuffer( l_context,
                                        CL_MEM_READ_ONLY, 
                                        sizeof(float)*l_n_values, 
                                        NULL, 
                                        &l_err );
    assert( l_err == CL_SUCCESS ); 

    cl_mem l_c_device = clCreateBuffer( l_context,
                                        CL_MEM_WRITE_ONLY, 
                                        sizeof(float)*l_n_values, 
                                        NULL, 
                                        &l_err );
    assert( l_err == CL_SUCCESS );

    // command queue
    cl_command_queue l_queue = clCreateCommandQueue(    l_context, 
                                                        l_device_ids[0], 
                                                        0, 
                                                        &l_err );

    // copy data from host to device
    std::cout << "copying data from host to device" << std::endl;
    l_err = clEnqueueWriteBuffer(   l_queue, 
                                    l_a_device,
                                    CL_TRUE,
                                    0,
                                    sizeof(float)*l_n_values,
                                    l_a_host,
                                    0,
                                    NULL,
                                    NULL );
    assert( l_err == CL_SUCCESS );

    l_err = clEnqueueWriteBuffer(   l_queue, 
                                    l_b_device,
                                    CL_TRUE,
                                    0,
                                    sizeof(float)*l_n_values,
                                    l_b_host,
                                    0,
                                    NULL,
                                    NULL );
    assert( l_err == CL_SUCCESS );

    // run kernel
    std::cout << "setting kernel parameters" << std::endl;
    l_err = clSetKernelArg( l_triad,
                            0,
                            sizeof(cl_mem),
                            &l_a_device );
    assert( l_err == CL_SUCCESS );

    l_err = clSetKernelArg( l_triad,
                            1,
                            sizeof(cl_mem),
                            &l_b_device );
    assert( l_err == CL_SUCCESS );

    l_err = clSetKernelArg( l_triad,
                            2,
                            sizeof(cl_mem),
                            &l_c_device );
    assert( l_err == CL_SUCCESS );
    
    std::cout << "running kernel" << std::endl;
    l_err = clEnqueueNDRangeKernel( l_queue,
                                    l_triad,
                                    1,
                                    NULL,
                                    &l_n_values,
                                    NULL,
                                    0,
                                    NULL,
                                    NULL);

    // wait for completion
    l_err = clFinish( l_queue );
    assert( l_err == CL_SUCCESS );

    std::cout << "successfully finished queue" << std::endl;

    // device host transfer
    std::cout << "copying data from device to host" << std::endl;
    l_err = clEnqueueReadBuffer(l_queue, 
                                l_c_device, 
                                CL_TRUE, 
                                0, 
                                sizeof(float)*l_n_values, 
                                l_c_host, 
                                0, 
                                NULL, 
                                NULL);     
    assert( l_err == CL_SUCCESS );

    std::cout << "printing result" << std::endl;
    for (std::size_t l_en = 0; l_en < l_n_values; l_en++)
    {
        std::cout << l_en << ": " << l_c_host[l_en] << std::endl;
    }



    delete [] l_a_host;
    delete [] l_b_host;
    delete [] l_c_host;

    delete [] l_device_ids;
    delete [] l_platform_ids;

    std::cout << "device query ended" << std::endl;
}