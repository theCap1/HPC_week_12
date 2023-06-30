#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else 
#include <CL/cl.h>
#endif

#include <cassert>
#include <iostream>

static const char * l_gemm = R"(
    __kernel void gemm( __global float4 * i_a,
                        __global float4 * i_b,
                        __global float4 * o_c,
                        __private uint l_m,
                        __private uint l_n,
                        __private uint l_k ){

        size_t l_gwid = get_global_id(0);
        size_t l_gwid_x = l_gwid%(l_n/8);
        size_t l_gwid_y = l_gwid/(l_n/8);
        size_t l_start_m = l_gwid_y*4;
        size_t l_start_n = l_gwid_x*2;
        size_t l_a_start = l_gwid_x*4*l_k/4;
        size_t l_b_start = l_gwid_x*8*l_k/4;
        size_t l_c_start = l_start_m*l_n/4+l_start_n;       // 8 vector blocking

        for(size_t m = 0; m < 4; m++){                      // m=4 in one block
            for(size_t n = 0; n < 2; n++){                  // n=8 in one block with 4 elements per vector
                for(size_t i = 0; i < l_k/4; i++){
                    o_c[l_c_start+m*l_n/4+n].w += dot(i_a[l_a_start+i+m*l_k/4], i_b[l_b_start+i+n*4*l_k/4]);
                    o_c[l_c_start+m*l_n/4+n].x += dot(i_a[l_a_start+i+m*l_k/4], i_b[l_b_start+i+n*4*l_k/4+1*l_k/4]);
                    o_c[l_c_start+m*l_n/4+n].y += dot(i_a[l_a_start+i+m*l_k/4], i_b[l_b_start+i+n*4*l_k/4+2*l_k/4]);
                    o_c[l_c_start+m*l_n/4+n].z += dot(i_a[l_a_start+i+m*l_k/4], i_b[l_b_start+i+n*4*l_k/4+3*l_k/4]);
                }  
            }
        }
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
    cl_context l_context = clCreateContext( NULL,
                                            1,
                                            l_device_ids+0,
                                            NULL,
                                            NULL,
                                            &l_err );
    assert( l_err == CL_SUCCESS );

    cl_program l_program = clCreateProgramWithSource(   l_context,
                                                        1,
                                                        &l_gemm,
                                                        NULL,
                                                        &l_err );
    assert( l_err == CL_SUCCESS );

    /*
     * build program 
     */
    std::cout << "build program: " << std::endl;
    std::cout << l_gemm << std::endl;
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
    cl_kernel l_gemm = clCreateKernel(  l_program,
                                        "gemm",
                                        &l_err );
    assert( l_err == CL_SUCCESS ); 

    // allocate  host memory
    std::cout << "allocating host memory" << std::endl;
    const std::size_t dataSize = 1;
    std::size_t l_m = dataSize*4;
    std::size_t l_n = dataSize*8;
    std::size_t l_k = dataSize*8;
    std::size_t global_work_size = dataSize*dataSize;   // how many threads? l_m/4*l_n/8 threads!
    
    cl_float4* l_a_host = new cl_float4[l_m*l_k/4];
    cl_float4* l_b_host = new cl_float4[l_n*l_k/4];
    cl_float4* l_c_host = new cl_float4[l_m*l_n/4];

    // initialize host memory
    std::cout << "initializing host memory" << std::endl;

    // array A[k]
    //         k -->
    //     |||---------------|----------------|||
    // m   |||0  4   8   12  |16   20  24  28 |||
    // |   |||---------------|----------------|||
    // v   |||1  5   9   13  |17   21  25  29 |||
    //     |||---------------|----------------|||
    //     |||2  6   10  14  |18   22  26  30 |||
    //     |||---------------|----------------|||
    //     |||3  7   11  15  |19   23  27  31 |||
    //     |||---------------|----------------|||
    for (std::size_t i = 0; i < l_m; i++)
    {
        for (std::size_t j = 0; j < l_k/4; j++)
        {
            l_a_host[i*l_k/4+j].w = j*4*l_m+i;
            l_a_host[i*l_k/4+j].x = j*4*l_m+i+l_m;
            l_a_host[i*l_k/4+j].y = j*4*l_m+i+2*l_m;
            l_a_host[i*l_k/4+j].z = j*4*l_m+i+3*l_m;
        }
    }
    std::cout << "initialization of A completed!" << std::endl;

    // test print array A
    std::cout << "print array A in float4 data type:" << std::endl;
    for (std::size_t i = 0; i < l_m; i++)
    {
        for (std::size_t j = 0; j < l_k/4; j++)
        {
            std::cout << l_a_host[i*l_k/4+j].w << "\t" << l_a_host[i*l_k/4+j].x << "\t" << l_a_host[i*l_k/4+j].y << "\t" << l_a_host[i*l_k/4+j].z << "\t";
        }
        std::cout << std::endl;
    }
    
    // std::cout << "printing of A completed!" << std::endl;

    // array B[k]
    //     n -->
    //     |||-  ||-  ||-  ||-  |||
    //     |||-  ||-  ||-  ||-  |||
    // k   |||0  ||24 ||48 ||72 |||
    // |   |||3  ||27 ||51 ||75 |||
    // v   |||6  ||30 ||54 ||78 |||
    //     |||9  ||33 ||57 ||81 |||
    //     |||-  ||-  ||-  ||-  |||
    //     |||12 ||36 ||60 ||84 |||
    //     |||15 ||39 ||63 ||87 |||
    //     |||18 ||42 ||66 ||90 |||
    //     |||21 ||45 ||69 ||93 |||
    //     |||-  ||-  ||-  ||-  |||

    for (std::size_t i = 0; i < l_n; i++)
    {
        for (std::size_t j = 0; j < l_k/4; j++)
        {
            l_b_host[i*l_k/4+j].w = (i*l_k+j*4);
            l_b_host[i*l_k/4+j].x = (i*l_k+j*4+1);
            l_b_host[i*l_k/4+j].y = (i*l_k+j*4+2);
            l_b_host[i*l_k/4+j].z = (i*l_k+j*4+3);
        }
    }
    std::cout << "initialization of B completed!" << std::endl;

    // test print array B
    std::cout << "print array B in float4 data type:" << std::endl;
    for (std::size_t i = 0; i < l_k/4; i++)
    {
        for (std::size_t j = 0; j < l_n; j++)
        {
            std::cout << l_b_host[i+j*l_k/4].w << "\t";
        }
        std::cout << std::endl;
        for (std::size_t j = 0; j < l_n; j++)
        {
            std::cout << l_b_host[i+j*l_k/4].x << "\t";
        }
        std::cout << std::endl;
        for (std::size_t j = 0; j < l_n; j++)
        {
            std::cout << l_b_host[i+j*l_k/4].y << "\t";
        }
        std::cout << std::endl;
        for (std::size_t j = 0; j < l_n; j++)
        {
            std::cout << l_b_host[i+j*l_k/4].z << "\t";
        }
        std::cout << std::endl;
    }

    // array C
    for (std::size_t i = 0; i < l_m*l_n/4; i++)
    {
        l_c_host[i].w = -1;
        l_c_host[i].x = -1;
        l_c_host[i].y = -1;
        l_c_host[i].z = -1;
    }
    std::cout << "initialization of C completed!" << std::endl;

    // test print array C
    // std::cout << "print array C in float4 data type:" << std::endl;
    // for (std::size_t i = 0; i < l_m; i++)
    // {
    //     for (std::size_t j = 0; j < l_n/4; j++)
    //     {
    //         std::cout << l_c_host[i*l_n/4+j].w << "\t" << l_c_host[i*l_n/4+j].x << "\t" << l_c_host[i*l_n/4+j].y << "\t" << l_c_host[i*l_n/4+j].z << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    std::cout << "allocation device memory" << std::endl;

    cl_mem l_a_device = clCreateBuffer( l_context,
                                        CL_MEM_READ_ONLY, 
                                        sizeof(cl_float4)*l_m*l_k/4, 
                                        NULL, 
                                        &l_err );
    assert( l_err == CL_SUCCESS ); 

    cl_mem l_b_device = clCreateBuffer( l_context,
                                        CL_MEM_READ_ONLY, 
                                        sizeof(cl_float4)*l_k/4*l_n, 
                                        NULL, 
                                        &l_err );
    assert( l_err == CL_SUCCESS ); 

    cl_mem l_c_device = clCreateBuffer( l_context,
                                        CL_MEM_WRITE_ONLY, 
                                        sizeof(cl_float4)*l_m*l_n/4, 
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
                                    sizeof(cl_float4)*l_k/4*l_m,
                                    l_a_host,
                                    0,
                                    NULL,
                                    NULL );
    assert( l_err == CL_SUCCESS );

    l_err = clEnqueueWriteBuffer(   l_queue, 
                                    l_b_device,
                                    CL_TRUE,
                                    0,
                                    sizeof(cl_float4)*l_k/4*l_n,
                                    l_b_host,
                                    0,
                                    NULL,
                                    NULL );
    assert( l_err == CL_SUCCESS );

    l_err = clEnqueueWriteBuffer(   l_queue, 
                                    l_c_device,
                                    CL_TRUE,
                                    0,
                                    sizeof(cl_float4)*l_n/4*l_m,
                                    l_c_host,
                                    0,
                                    NULL,
                                    NULL );
    assert( l_err == CL_SUCCESS );


    // run kernel
    std::cout << "setting kernel parameters" << std::endl;
    l_err = clSetKernelArg( l_gemm,
                            0,
                            sizeof(cl_mem),
                            &l_a_device );
    assert( l_err == CL_SUCCESS );

    l_err = clSetKernelArg( l_gemm,
                            1,
                            sizeof(cl_mem),
                            &l_b_device );
    assert( l_err == CL_SUCCESS );

    l_err = clSetKernelArg( l_gemm,
                            2,
                            sizeof(cl_mem),
                            &l_c_device );
    assert( l_err == CL_SUCCESS );

    cl_uint l_tmp_uint = static_cast<cl_uint>(l_m);
    l_err = clSetKernelArg( l_gemm,
                            3,
                            sizeof(cl_uint),
                            &l_tmp_uint );
    assert( l_err == CL_SUCCESS );

    l_tmp_uint = static_cast<cl_uint>(l_n);
    l_err = clSetKernelArg( l_gemm,
                            4,
                            sizeof(cl_uint),
                            &l_tmp_uint );
    assert( l_err == CL_SUCCESS );

    l_tmp_uint = static_cast<cl_uint>(l_k);
    l_err = clSetKernelArg( l_gemm,
                            5,
                            sizeof(cl_uint),
                            &l_tmp_uint );
    assert( l_err == CL_SUCCESS );
    
    std::cout << "running kernel" << std::endl;
    l_err = clEnqueueNDRangeKernel( l_queue,
                                    l_gemm,
                                    1,
                                    NULL,
                                    &global_work_size,
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
                                sizeof(cl_float4)*l_n/4*l_m, 
                                l_c_host, 
                                0, 
                                NULL, 
                                NULL);     
    assert( l_err == CL_SUCCESS );

    // wait for completion
    l_err = clFinish( l_queue );
    assert( l_err == CL_SUCCESS );

    std::cout << "printing result" << std::endl;
    for (std::size_t i = 0; i < l_m; i++)
    {
        for (std::size_t j = 0; j < l_n/4; j++)
        {
            std::cout << l_c_host[i*l_n/4+j].w << "\t" << l_c_host[i*l_n/4+j].x << "\t" << l_c_host[i*l_n/4+j].y << "\t" << l_c_host[i*l_n/4+j].z << "\t";
        }
        std::cout << std::endl;
    }


    delete [] l_a_host;
    delete [] l_b_host;
    delete [] l_c_host;

    delete [] l_device_ids;
    delete [] l_platform_ids;

    std::cout << "device query ended" << std::endl;
}