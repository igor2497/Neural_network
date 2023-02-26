#define _CRT_SECURE_NO_WARNINGS
#include<stdlib.h>
#include<iostream>
#include<fstream>
#include"nn.h"
#include"quaternion.h"
#include"stick_balancing.h"
#include"thread_pool.h"
#define _USE_MATH_DEFINES
#include<math.h>
#include<thread> 
#include<ctime>
#include<string>
#include<mutex>

// GUI
#include"imgui.h"
#include"imgui_impl_win32.h"
#include"imgui_impl_dx11.h"
#include<d3d11.h>
#include<tchar.h>

//#define BENCHMARK

#ifdef BENCHMARK
#define MAX_GENERATIONS     10                      //!< Maximum number of generations that simulation will run
#endif

#if population < next_gen
#error next generation can not be greater than population
#endif // #if population < next_gen

using namespace std;

typedef struct GUI_parameters {
    bool stop;                          //!< Flag for stopping the simulation
    bool newGenerationDone;             //!< Refers to: worstInGeneration, bestInGeneration, generationAverage
    unsigned int worstInGeneration;
    unsigned int bestInGeneration;
    float generationAverage;
    unsigned int generation;
    int mutationSlider;
#ifdef GPU_USE
    bool saveGpuGen;
#else
    bool saveCpuGen;
#endif // #ifdef GPU_USE
} GUI_param;

typedef struct threadParameters {
    Neural_network *nn;
    Stick pendulum;
    Stick pendulum2;
    Stick pendulum3;
    unsigned int result;
    bool should_save_best;
#ifdef GPU_USE
    openCl_param *gpuParam;
#endif // #ifdef GPU_USE
} threadParam;

static void tasks_done_handler();

void thread_pool(threadParam *param);

void renderGUI(GUI_param *param);

volatile bool tasksDone = false;

// Data
static ID3D11Device*            g_pd3dDevice = NULL;
static ID3D11DeviceContext*     g_pd3dDeviceContext = NULL;
static IDXGISwapChain*          g_pSwapChain = NULL;
static ID3D11RenderTargetView*  g_mainRenderTargetView = NULL;

// Forward declarations of helper functions
bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

void main() {
    unsigned int i, best_id[next_gen], roulette[roulette_size], tc, j, results[population];
    unsigned int rows, columns, current_best = 0, previous_best = 0, generation = 0;
    int current_rank = 0;
    float axisVector[eulerCount] = { -1, 1, -1 };
    Quaternion pendulumQ(PENDULUM_X_ANGLE, 0, PENDULUM_Z_ANGLE, 1);
    Quaternion pendulumQ2(0, 0, 0, 1), pendulumQ3(0, 0, 0, 1);
    double pSize[eulerCount] = { 0.1, PENDULUM_HEIGHT, 0.1 };
    double pPosition[eulerCount] = { 0, PENDULUM_Y_POS, 0 };
    double pVelocity[eulerCount] = { 0, 0, 0 };
    double pAngularV[eulerCount] = { 0, 0, 0 };
    double pSize2[eulerCount] = { 0.1, PENDULUM_HEIGHT * 1.2, 0.1 };
    double pPosition2[eulerCount] = { 0, PENDULUM_Y_POS, 0 };
    double pVelocity2[eulerCount] = { 0, 0, 0 };
    double pAngularV2[eulerCount] = { 0, 0, 0 };
    double pSize3[eulerCount] = { 0.1, PENDULUM_HEIGHT * 1.4, 0.1 };
    double pPosition3[eulerCount] = { 0, PENDULUM_Y_POS, 0 };
    double pVelocity3[eulerCount] = { 0, 0, 0 };
    double pAngularV3[eulerCount] = { 0, 0, 0 };
    double pForce[eulerCount] = { 0, 0, 0 };
    double jointForce[eulerCount] = { 0, 0, 0 };
    double jointForce23[eulerCount] = { 0, 0, 0 };
    double zeroForce[eulerCount] = { 0, 0, 0 };
    double best[next_gen], roulette_sum = 0, roulette_odds[next_gen];
    bool pass = false;
    string saveFile;
    ERR_E  errLocal = ERR_OK;
    long int scoreSum = 0;
    float averageScore;
    int mutationTemp = mutation;
    thread_pool_param threadPoolParam;
#ifdef BENCHMARK
    string loadFile;
#endif // #ifdef BENCHMARK
    clock_t startTime;

    threadParam *threadParameter = (threadParam *)malloc(population * sizeof(threadParam));
    /*{ nullptr, pendulum , pendulum2, pendulum3, 0, 0,
#ifdef GPU_USE
        nullptr,
#endif // #ifdef GPU_USE
        true };*/

    char *unityRotation = "C:/repo/unity/inverted pendulum/Assets/angles.txt";
    char *unityPosition = "C:/repo/unity/inverted pendulum/Assets/position.txt";
    char *unityRotation2 = "C:/repo/unity/inverted pendulum/Assets/angles2.txt";
    char *unityPosition2 = "C:/repo/unity/inverted pendulum/Assets/position2.txt";
    char *unityRotation3 = "C:/repo/unity/inverted pendulum/Assets/angles3.txt";
    char *unityPosition3 = "C:/repo/unity/inverted pendulum/Assets/position3.txt";
    ofstream qFile, pFile, qFile2, pFile2, qFile3, pFile3;

    Neural_network *nn[population], *temp_nn[population];
    thread gui_thread;

    GUI_param guiParam = { false, false, 0, 0, 0, 0, mutation, false };


    srand((int)time(NULL) + (int)clock());

    gui_thread = thread(renderGUI, &guiParam);


    startTime = clock();

    // create neural networks
    for(i = 0; i < population; i++) {
#ifndef BENCHMARK
#if hidden_layers == 3
        nn[i] = new Neural_network(input_layer, hidden1, hidden2, hidden3, output_layer);
        temp_nn[i] = new Neural_network(input_layer, hidden1, hidden2, hidden3, output_layer);
#elif hidden_layers == 2
        nn[i] = new Neural_network(input_layer, hidden1, hidden2, output_layer);
        temp_nn[i] = new Neural_network(input_layer, hidden1, hidden2, output_layer);
#elif hidden_layers == 1
        nn[i] = new Neural_network(input_layer, hidden1, output_layer);
        temp_nn[i] = new Neural_network(input_layer, hidden1, output_layer);
        best_nn = new Neural_network(input_layer, hidden1, output_layer);
#endif // #if hidden_layers == 3
#else
#ifdef GPU_USE
        loadFile = "gpu_benchmark/benchmark" + to_string(i) + ".bin";
#else
        loadFile = "cpu_benchmark/benchmark" + to_string(i) + ".bin";
#endif // #ifdef GPU_USE

        nn[i] = new Neural_network(loadFile.c_str());
#if hidden_layers == 3
        temp_nn[i] = new Neural_network(input_layer, hidden1, hidden2, hidden3, output_layer);
#elif hidden_layers == 2
        temp_nn[i] = new Neural_network(input_layer, hidden1, hidden2, output_layer);
#elif hidden_layers == 1
        temp_nn[i] = new Neural_network(input_layer, hidden1, output_layer);
#endif // #if hidden_layers == 3
#endif // #ifndef BENCHMARK
    }

    // calculate odds for each neural network to get mixed in new generation
    for(i = 0; i < next_gen; i++) {
        roulette_sum += 1.0 / ((double)i + 2.0);
        roulette_odds[i] = roulette_sum;
    }

    // set an array to bias better networks
    for(i = 0; i < roulette_size; i++) {
        roulette[i] = current_rank;
        if(i > roulette_odds[current_rank] / roulette_sum * roulette_size) {
            if(current_rank < next_gen - 1) {
                current_rank++;
            }
        }
    }

    pendulumQ.normalize();

    Stick pendulum(PENDULUM_MASS,
        pSize,
        pPosition,
        pendulumQ,
        pVelocity,
        pAngularV);

    Stick pendulum2(PENDULUM2_MASS,
        pSize2,
        pPosition2,
        pendulumQ2,
        pVelocity2,
        pAngularV2);

    Stick pendulum3(PENDULUM3_MASS,
        pSize3,
        pPosition3,
        pendulumQ3,
        pVelocity3,
        pAngularV3);

    // adjust pendulum position
    pendulum.position[eulerX] = pendulum.massVector.rotate(pendulum.rotation).x;
    pendulum.position[eulerY] = pendulum.massVector.rotate(pendulum.rotation).y;
    pendulum.position[eulerZ] = pendulum.massVector.rotate(pendulum.rotation).z;

    pendulum2.position[eulerX] = pendulum.massVector.rotate(pendulum.rotation).x * 2;
    pendulum2.position[eulerY] = pendulum.massVector.rotate(pendulum.rotation).y * 2 + pendulum2.massVector.y;
    pendulum2.position[eulerZ] = pendulum.massVector.rotate(pendulum.rotation).z * 2;

    pendulum3.position[eulerX] = pendulum2.position[eulerX];
    pendulum3.position[eulerY] = pendulum2.position[eulerY] + pendulum2.massVector.y + pendulum3.massVector.y;
    pendulum3.position[eulerZ] = pendulum2.position[eulerZ];

    thread_pool_param threadInitParam;
    threadInitParam._poolSize = population;
    threadInitParam._task = &thread_pool;
    threadInitParam._parameters = threadParameter;
    threadInitParam.callback = &tasks_done_handler;
    threadInitParam._taskNumber = 0;
    threadInitParam._paramSize = sizeof(threadParam);
#ifdef GPU_USE
    threadInitParam.clFileName = "nn.cl";
#endif // #ifdef GPU_USE

    thread_pool_init(&threadInitParam, &errLocal);

    cout << "1st interval: " << (float)(clock() - startTime) / 1000 << endl;

    // run MAX_GENERATIONS generations
    while((generation < MAX_GENERATIONS) && (!guiParam.stop)) {
        cout << "Generation " << generation + 1 << endl;

        mutationTemp = guiParam.mutationSlider;

#ifndef BENCHMARK
        // new random position of a stick for every generation
        pendulum.rotation.x = (double)((double)rand() / RAND_MAX * 2 * PENDULUM_X_ANGLE - PENDULUM_X_ANGLE);
        pendulum.rotation.z = (double)((double)rand() / RAND_MAX * 2 * PENDULUM_Z_ANGLE - PENDULUM_Z_ANGLE);
#endif // #ifndef BENCHMARK
        pendulum.position[eulerY] = pendulum.massVector.rotate(pendulum.rotation).y;
        pendulum.position[eulerX] = pendulum.massVector.rotate(pendulum.rotation).x;
        pendulum.position[eulerZ] = pendulum.massVector.rotate(pendulum.rotation).z;

        pendulum2.position[eulerX] = pendulum.massVector.rotate(pendulum.rotation).x * 2;
        pendulum2.position[eulerY] = pendulum.massVector.rotate(pendulum.rotation).y * 2 + pendulum2.massVector.y;
        pendulum2.position[eulerZ] = pendulum.massVector.rotate(pendulum.rotation).z * 2;

        pendulum3.position[eulerX] = pendulum2.position[eulerX];
        pendulum3.position[eulerY] = pendulum2.position[eulerY] + pendulum2.massVector.y + pendulum3.massVector.y;
        pendulum3.position[eulerZ] = pendulum2.position[eulerZ];

        // initialize the arrays that hold best results and IDs
        for(i = 0; i < next_gen; i++) {
            best[i] = 0;
            best_id[i] = population;
        }

        for(tc = 0; tc < population; tc++) {
            threadParameter[tc].nn = nn[tc];
            threadParameter[tc].pendulum = pendulum;
            threadParameter[tc].pendulum2 = pendulum2;
            threadParameter[tc].pendulum3 = pendulum3;
            threadParameter[tc].result = 0;
            threadParameter[tc].should_save_best = false;
        }
        
        tasksDone = false;

        thread_pool_start(&errLocal);

        while(!tasksDone);

        // initialise the arrays
        for(i = 0; i < next_gen; i++) {
            best[i] = threadParameter[i].result;
            best_id[i] = i;
        }

        // sort the results
        for(i = 0; i < next_gen; i++) {
            for(tc = 0; tc < population; tc++) {
                if(threadParameter[tc].result > best[i]) {
                    for(j = 0; j < i; j++) {
                        if(tc == best_id[j]) {
                            pass = true;
                        }
                    }
                    if(!pass) {
                        best[i] = threadParameter[tc].result;
                        best_id[i] = tc;
                    }
                    pass = false;
                }
            }
        }

        scoreSum = 0;

        for(i = 0; i < population; i++) {
            scoreSum += threadParameter[i].result;
        }

        averageScore = (double)scoreSum / (double)population;
        guiParam.generationAverage = averageScore;
        guiParam.bestInGeneration = threadParameter[best_id[0]].result;
        guiParam.worstInGeneration = best[next_gen - 1];
        guiParam.generation = generation;
        guiParam.newGenerationDone = true;

        // generate the next generation of neural networks
#if hidden_layers == 2
        for(i = 0; i < population; i++) {
            for(rows = 0; rows < temp_nn[i]->ho->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->ho->col; columns++) {
                    temp_nn[i]->ho->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->ho->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->h12->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->h12->col; columns++) {
                    temp_nn[i]->h12->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->h12->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->ih1->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->ih1->col; columns++) {
                    temp_nn[i]->ih1->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->ih1->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->ob->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->ob->col; columns++) {
                    temp_nn[i]->ob->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->ob->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->b1->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->b1->col; columns++) {
                    temp_nn[i]->b1->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->b1->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->b2->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->b2->col; columns++) {
                    temp_nn[i]->b2->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->b2->getel(rows, columns));
                }
            }
        }
#elif hidden_layers == 3
        for(i = 0; i < population; i++) {
            for(rows = 0; rows < temp_nn[i]->ho->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->ho->col; columns++) {
                    temp_nn[i]->ho->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->ho->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->h12->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->h12->col; columns++) {
                    temp_nn[i]->h12->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->h12->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->h23->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->h23->col; columns++) {
                    temp_nn[i]->h23->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->h23->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->ih1->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->ih1->col; columns++) {
                    temp_nn[i]->ih1->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->ih1->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->ob->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->ob->col; columns++) {
                    temp_nn[i]->ob->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->ob->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->b1->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->b1->col; columns++) {
                    temp_nn[i]->b1->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->b1->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->b2->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->b2->col; columns++) {
                    temp_nn[i]->b2->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->b2->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->b3->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->b3->col; columns++) {
                    temp_nn[i]->b3->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->b3->getel(rows, columns));
                }
            }
        }
#elif hidden_layers == 1
        for(i = 0; i < population; i++) {
            for(rows = 0; rows < temp_nn[i]->ho->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->ho->col; columns++) {
                    temp_nn[i]->ho->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->ho->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->ih1->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->ih1->col; columns++) {
                    temp_nn[i]->ih1->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->ih1->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->ob->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->ob->col; columns++) {
                    temp_nn[i]->ob->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->ob->getel(rows, columns));
                }
            }
            for(rows = 0; rows < temp_nn[i]->b1->row; rows++) {
                for(columns = 0; columns < temp_nn[i]->b1->col; columns++) {
                    temp_nn[i]->b1->setel(rows, columns, nn[best_id[roulette[rand() % roulette_size]]]->b1->getel(rows, columns));
                }
            }
        }
#endif

#ifndef BENCHMARK
#ifdef GPU_USE
        if(guiParam.saveGpuGen) {
            for(i = 0; i < population; i++) {
                saveFile = "gpu_benchmark/benchmark" + to_string(i) + ".bin";

                nn[i]->save(saveFile.c_str(), &errLocal);
            }

            guiParam.saveGpuGen = false;
        }
#else
        if(guiParam.saveCpuGen) {
            for(i = 0; i < population; i++) {
                saveFile = "cpu_benchmark/benchmark" + to_string(i) + ".bin";

                nn[i]->save(saveFile.c_str(), &errLocal);
            }
            guiParam.saveCpuGen = false;
        }
#endif // #ifdef GPU_USE
        
#endif // #ifndef BENCHMARK

        // check if it is the new session record
        if(threadParameter[best_id[0]].result >= current_best) {

            cout << "--------------Best in session: " << threadParameter[best_id[0]].result;

            cout << "\tGeneration average: " << averageScore << "\tWorst in new generation: " << best[next_gen - 1] << endl;

            // set the new best score
            current_best = threadParameter[best_id[0]].result;

            saveFile = "bis" + to_string(generation + 1) + ".bin";

            nn[best_id[0]]->save(saveFile.c_str(), &errLocal);

            threadParameter[best_id[0]].result = 0;
            threadParameter[best_id[0]].should_save_best = true;
            
            thread_pool(&(threadParameter[best_id[0]]));

        } else {
            cout << "Best in generation: " << threadParameter[best_id[0]].result;
            cout << "\t\t\tGeneration average: " << averageScore << "\tWorst in new generation: " << best[next_gen - 1] << endl;
        }

#ifndef BENCHMARK
        // apply the new weights and add mutation
        if(threadParameter[best_id[0]].result > previous_best) {

            // if it is new best, make one exact copy of the best neural network
            nn[0]->copy(*nn[best_id[0]]);

            // "randomize" the others
            for(i = 1; i < population; i++) {
                nn[i]->copy(*temp_nn[i]);
                nn[i]->randomize(mutationTemp);
            }
            previous_best = current_best;
        } else {

            // "randomize" the next generation
            for(i = 0; i < population; i++) {
                nn[i]->copy(*(temp_nn[i]));
                nn[i]->randomize(mutationTemp);
            }
        }
#endif // #ifndef BENCHMARK

        generation++;
    } // while (generation < 1000)

    thread_pool_stop(&errLocal);

    cout << "Time: " << (float)(clock() - startTime)/1000 << endl;

    cout << "Done" << endl;

    gui_thread.join();
}





void thread_pool(threadParam *threadParams) {
    Matrix inputs(input_layer, 1, false);
    double pendulum_height = threadParams->pendulum.massVector.rotate(threadParams->pendulum.rotation).y;
    double pForce[eulerCount];
    double jointForce[eulerCount] = { 0, 0, 0 };
    double jointForce23[eulerCount] = { 0, 0, 0 };
    double zeroForce[eulerCount] = { 0, 0, 0 };
    char *unityRotation = "C:/repo/unity/inverted pendulum/Assets/angles.txt";
    char *unityPosition = "C:/repo/unity/inverted pendulum/Assets/position.txt";
    char *unityRotation2 = "C:/repo/unity/inverted pendulum/Assets/angles2.txt";
    char *unityPosition2 = "C:/repo/unity/inverted pendulum/Assets/position2.txt";
    char *unityRotation3 = "C:/repo/unity/inverted pendulum/Assets/angles3.txt";
    char *unityPosition3 = "C:/repo/unity/inverted pendulum/Assets/position3.txt";
    ofstream qFile, pFile, qFile2, pFile2, qFile3, pFile3;
    float eulerRotation[eulerCount];
#ifdef GPU_USE
    double GPU_results[output_layer];
    // OpenCl
    cl_int ret = 0;
#else
    Matrix outputs(output_layer, 1, false);
#endif // #ifdef GPU_USE

    // if the run is being recorded
    if(threadParams->should_save_best) {
        // Delete old file and create new
        remove(unityRotation);
        qFile.open(unityRotation);
        remove(unityPosition);
        pFile.open(unityPosition);

        remove(unityRotation2);
        qFile2.open(unityRotation2);
        remove(unityPosition2);
        pFile2.open(unityPosition2);

        remove(unityRotation3);
        qFile3.open(unityRotation3);
        remove(unityPosition2);
        pFile3.open(unityPosition3);
    }

#ifdef GPU_USE
    // Copying array to buffer
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->ih1_mem_obj, CL_TRUE, 0, threadParams->nn->il * threadParams->nn->hl1 * sizeof(double), threadParams->nn->ih1->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->b1_mem_obj, CL_TRUE, 0, threadParams->nn->hl1 * sizeof(double), threadParams->nn->b1->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->columns_mem_obj, CL_TRUE, 0, sizeof(int), &(threadParams->nn->il), 0, NULL, NULL);
#if hidden_layers > 1
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->hl2_mem_obj, CL_TRUE, 0, threadParams->nn->hl1 * threadParams->nn->hl2 * sizeof(double), threadParams->nn->h12->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->b2_mem_obj, CL_TRUE, 0, threadParams->nn->hl2 * sizeof(double), threadParams->nn->b2->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->columns_mem_obj2, CL_TRUE, 0, sizeof(int), &(threadParams->nn->hl1), 0, NULL, NULL);
#if hidden_layers > 2
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->hl3_mem_obj, CL_TRUE, 0, threadParams->nn->hl2 * threadParams->nn->hl3 * sizeof(double), threadParams->nn->h23->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->b3_mem_obj, CL_TRUE, 0, threadParams->nn->hl3 * sizeof(double), threadParams->nn->b3->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->columns_mem_obj3, CL_TRUE, 0, sizeof(int), &(threadParams->nn->hl2), 0, NULL, NULL);

    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->ol_mem_obj, CL_TRUE, 0, threadParams->nn->hl3 * threadParams->nn->ol * sizeof(double), threadParams->nn->ho->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->ob_mem_obj, CL_TRUE, 0, threadParams->nn->ol * sizeof(double), threadParams->nn->ob->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->columns_mem_obj_out, CL_TRUE, 0, sizeof(int), &(threadParams->nn->hl3), 0, NULL, NULL);
#else // #if hidden_layers > 2
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->ol_mem_obj, CL_TRUE, 0, threadParams->nn->hl2 * threadParams->nn->ol * sizeof(double), threadParams->nn->ho->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->ob_mem_obj, CL_TRUE, 0, threadParams->nn->ol * sizeof(double), threadParams->nn->ob->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->columns_mem_obj_out, CL_TRUE, 0, sizeof(int), &(threadParams->nn->hl2), 0, NULL, NULL);
#endif // #if hidden_layers > 2
#else // #if hidden_layers > 1
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->ol_mem_obj, CL_TRUE, 0, threadParams->nn->hl1 * threadParams->nn->ol * sizeof(double), threadParams->nn->ho->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->ob_mem_obj, CL_TRUE, 0, threadParams->nn->ol * sizeof(double), threadParams->nn->ob->matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->columns_mem_obj_out, CL_TRUE, 0, sizeof(int), &(threadParams->nn->hl1), 0, NULL, NULL);
#endif // #if hidden_layers > 1
#endif // #ifdef GPU_USE

    // run until the simulation timeouts
    while(threadParams->result < SIM_TIMEOUT) {
        // calculate the stick euler angle to feed it as input
        threadParams->pendulum.rotation.quaternionToEuler(eulerRotation);
        inputs.setel(0, 0, eulerRotation[eulerX]);
        inputs.setel(1, 0, eulerRotation[eulerZ]);
        // angular velocity
        inputs.setel(2, 0, threadParams->pendulum.angularV[eulerX]);
        inputs.setel(3, 0, threadParams->pendulum.angularV[eulerZ]);
        // position
        inputs.setel(4, 0, threadParams->pendulum.position[eulerX]);
        inputs.setel(5, 0, threadParams->pendulum.position[eulerZ]);

        threadParams->pendulum2.rotation.quaternionToEuler(eulerRotation);
        inputs.setel(6, 0, eulerRotation[eulerX]);
        inputs.setel(7, 0, eulerRotation[eulerZ]);

        inputs.setel(8, 0, threadParams->pendulum2.angularV[eulerX]);
        inputs.setel(9, 0, threadParams->pendulum2.angularV[eulerZ]);

        threadParams->pendulum3.rotation.quaternionToEuler(eulerRotation);
        inputs.setel(10, 0, eulerRotation[eulerX]);
        inputs.setel(11, 0, eulerRotation[eulerZ]);

        inputs.setel(12, 0, threadParams->pendulum3.angularV[eulerX]);
        inputs.setel(13, 0, threadParams->pendulum3.angularV[eulerZ]);

        // calculate the neural network output
#ifdef GPU_USE
        ret = clEnqueueWriteBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->inputs_mem_obj, CL_TRUE, 0, input_layer * sizeof(double), inputs.matrix, 0, NULL, NULL);

        ret = clEnqueueNDRangeKernel(threadParams->gpuParam->command_queue, threadParams->gpuParam->kernel, 1, NULL, &(threadParams->gpuParam->global_item_size), &(threadParams->gpuParam->local_item_size), 0, NULL, NULL);
#if hidden_layers > 1
        ret = clEnqueueNDRangeKernel(threadParams->gpuParam->command_queue, threadParams->gpuParam->kernel2, 1, NULL, &(threadParams->gpuParam->global_item_size2), &(threadParams->gpuParam->local_item_size2), 0, NULL, NULL);
#if hidden_layers > 2
        ret = clEnqueueNDRangeKernel(threadParams->gpuParam->command_queue, threadParams->gpuParam->kernel3, 1, NULL, &(threadParams->gpuParam->global_item_size3), &(threadParams->gpuParam->local_item_size3), 0, NULL, NULL);
#endif // #if hidden_layers > 2
#endif // #if hidden_layers > 1
        ret = clEnqueueNDRangeKernel(threadParams->gpuParam->command_queue, threadParams->gpuParam->kernelOut, 1, NULL, &(threadParams->gpuParam->global_item_size_out), &(threadParams->gpuParam->local_item_size_out), 0, NULL, NULL);

        ret = clEnqueueReadBuffer(threadParams->gpuParam->command_queue, threadParams->gpuParam->result_mem_obj, CL_TRUE, 0, output_layer * sizeof(double), GPU_results, 0, NULL, NULL);
#else
        outputs.setmat(threadParams->nn->calculate(inputs));
#endif // #ifdef GPU_USE

        // set the horizontal forces from network output and calculate the vertical force to support the stick
#ifdef GPU_USE
        pForce[eulerX] = (GPU_results[0] - 0.5) * threadParams->pendulum.mass * gravity * AI_FORCE * 2;
        pForce[eulerZ] = (GPU_results[1] - 0.5) * threadParams->pendulum.mass * gravity * AI_FORCE * 2;
#else
        pForce[eulerX] = (outputs.getel(0, 0) - 0.5) * threadParams->pendulum.mass * gravity * AI_FORCE * 2;
        pForce[eulerZ] = (outputs.getel(1, 0) - 0.5) * threadParams->pendulum.mass * gravity * AI_FORCE * 2;
#endif // #ifdef GPU_USE
        pForce[eulerY] = -((threadParams->pendulum.massVector.conjugate().rotate(threadParams->pendulum.rotation)).y + threadParams->pendulum.position[eulerY]) / TIME_STEP / TIME_STEP * threadParams->pendulum.mass;
        // reduce the vertical force if it is to big
        if(pForce[eulerY] < -threadParams->pendulum.mass * gravity * FORCE_CLIPOFF) {
            pForce[eulerY] = -threadParams->pendulum.mass * gravity * FORCE_CLIPOFF;
        }
        if(pForce[eulerY] > threadParams->pendulum.mass * gravity * FORCE_CLIPOFF) {
            pForce[eulerY] = threadParams->pendulum.mass * gravity * FORCE_CLIPOFF;
        }

        // joint force 1-2
        jointForce[eulerY] = (((threadParams->pendulum.massVector.rotate(threadParams->pendulum.rotation)).y + threadParams->pendulum.position[eulerY]) -
            ((threadParams->pendulum2.massVector.conjugate().rotate(threadParams->pendulum2.rotation)).y + threadParams->pendulum2.position[eulerY]))
            / TIME_STEP / TIME_STEP * threadParams->pendulum.mass / NUM_OF_BODIES;
        jointForce[eulerX] = (((threadParams->pendulum.massVector.rotate(threadParams->pendulum.rotation)).x + threadParams->pendulum.position[eulerX]) -
            ((threadParams->pendulum2.massVector.conjugate().rotate(threadParams->pendulum2.rotation)).x + threadParams->pendulum2.position[eulerX]))
            / TIME_STEP / TIME_STEP * threadParams->pendulum.mass / NUM_OF_BODIES;
        jointForce[eulerZ] = (((threadParams->pendulum.massVector.rotate(threadParams->pendulum.rotation)).z + threadParams->pendulum.position[eulerZ]) -
            ((threadParams->pendulum2.massVector.conjugate().rotate(threadParams->pendulum2.rotation)).z + threadParams->pendulum2.position[eulerZ]))
            / TIME_STEP / TIME_STEP * threadParams->pendulum.mass / NUM_OF_BODIES;

        // reduce the force if it is to big
        if(jointForce[eulerY] < -threadParams->pendulum.mass * gravity * FORCE_CLIPOFF) {
            jointForce[eulerY] = -threadParams->pendulum.mass * gravity * FORCE_CLIPOFF;
        }
        if(jointForce[eulerY] > threadParams->pendulum.mass * gravity * FORCE_CLIPOFF) {
            jointForce[eulerY] = threadParams->pendulum.mass * gravity * FORCE_CLIPOFF;
        }

        if(jointForce[eulerX] > threadParams->pendulum.mass * gravity * FORCE_CLIPOFF) {
            jointForce[eulerX] = threadParams->pendulum.mass * gravity * FORCE_CLIPOFF;
        }
        if(jointForce[eulerX] < -threadParams->pendulum.mass * gravity * FORCE_CLIPOFF) {
            jointForce[eulerX] = -threadParams->pendulum.mass * gravity * FORCE_CLIPOFF;
        }

        if(jointForce[eulerZ] > threadParams->pendulum.mass * gravity * FORCE_CLIPOFF) {
            jointForce[eulerZ] = threadParams->pendulum.mass * gravity * FORCE_CLIPOFF;
        }
        if(jointForce[eulerZ] < -threadParams->pendulum.mass * gravity * FORCE_CLIPOFF) {
            jointForce[eulerZ] = -threadParams->pendulum.mass * gravity * FORCE_CLIPOFF;
        }

        // joint force 3-4
        jointForce23[eulerY] = (((threadParams->pendulum2.massVector.rotate(threadParams->pendulum2.rotation)).y + threadParams->pendulum2.position[eulerY]) -
            ((threadParams->pendulum3.massVector.conjugate().rotate(threadParams->pendulum3.rotation)).y + threadParams->pendulum3.position[eulerY]))
            / TIME_STEP / TIME_STEP * threadParams->pendulum2.mass / NUM_OF_BODIES;
        jointForce23[eulerX] = (((threadParams->pendulum2.massVector.rotate(threadParams->pendulum2.rotation)).x + threadParams->pendulum2.position[eulerX]) -
            ((threadParams->pendulum3.massVector.conjugate().rotate(threadParams->pendulum3.rotation)).x + threadParams->pendulum3.position[eulerX]))
            / TIME_STEP / TIME_STEP * threadParams->pendulum2.mass / NUM_OF_BODIES;
        jointForce23[eulerZ] = (((threadParams->pendulum2.massVector.rotate(threadParams->pendulum2.rotation)).z + threadParams->pendulum2.position[eulerZ]) -
            ((threadParams->pendulum3.massVector.conjugate().rotate(threadParams->pendulum3.rotation)).z + threadParams->pendulum3.position[eulerZ]))
            / TIME_STEP / TIME_STEP * threadParams->pendulum2.mass / NUM_OF_BODIES;

        // reduce the force if it is to big
        if(jointForce23[eulerY] < -threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF) {
            jointForce23[eulerY] = -threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF;
        }
        if(jointForce23[eulerY] > threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF) {
            jointForce23[eulerY] = threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF;
        }

        if(jointForce23[eulerX] > threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF) {
            jointForce23[eulerX] = threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF;
        }
        if(jointForce23[eulerX] < -threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF) {
            jointForce23[eulerX] = -threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF;
        }

        if(jointForce23[eulerZ] > threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF) {
            jointForce23[eulerZ] = threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF;
        }
        if(jointForce23[eulerZ] < -threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF) {
            jointForce23[eulerZ] = -threadParams->pendulum2.mass * gravity * FORCE_CLIPOFF;
        }

        // calculate the stick physics
        threadParams->pendulum3.physics(jointForce23, zeroForce, TIME_STEP);

        // flip the force for pendulum2 because it has opposite reaction from top one
        jointForce23[eulerX] = -jointForce23[eulerX];
        jointForce23[eulerY] = -jointForce23[eulerY];
        jointForce23[eulerZ] = -jointForce23[eulerZ];

        threadParams->pendulum2.physics(jointForce, jointForce23, TIME_STEP);

        // flip the force for bottom pendulum because it has opposite reaction from top one
        jointForce[eulerX] = -jointForce[eulerX];
        jointForce[eulerY] = -jointForce[eulerY];
        jointForce[eulerZ] = -jointForce[eulerZ];

        threadParams->pendulum.physics(pForce, jointForce, TIME_STEP);

        // if the run is being recorded
        if(threadParams->should_save_best) {
            // Write the stick parameters to file
            qFile << threadParams->pendulum.rotation.x;
            qFile << "\n";
            qFile << threadParams->pendulum.rotation.y;
            qFile << "\n";
            qFile << threadParams->pendulum.rotation.z;
            qFile << "\n";
            qFile << threadParams->pendulum.rotation.w;
            qFile << "\n";

            pFile << threadParams->pendulum.position[eulerX];
            pFile << "\n";
            pFile << threadParams->pendulum.position[eulerY];
            pFile << "\n";
            pFile << threadParams->pendulum.position[eulerZ];
            pFile << "\n";

            // Write the middle pendulum parameters to file
            qFile2 << threadParams->pendulum2.rotation.x;
            qFile2 << "\n";
            qFile2 << threadParams->pendulum2.rotation.y;
            qFile2 << "\n";
            qFile2 << threadParams->pendulum2.rotation.z;
            qFile2 << "\n";
            qFile2 << threadParams->pendulum2.rotation.w;
            qFile2 << "\n";

            pFile2 << threadParams->pendulum2.position[eulerX];
            pFile2 << "\n";
            pFile2 << threadParams->pendulum2.position[eulerY];
            pFile2 << "\n";
            pFile2 << threadParams->pendulum2.position[eulerZ];
            pFile2 << "\n";

            // Write the top pendulum parameters to file
            qFile3 << threadParams->pendulum3.rotation.x;
            qFile3 << "\n";
            qFile3 << threadParams->pendulum3.rotation.y;
            qFile3 << "\n";
            qFile3 << threadParams->pendulum3.rotation.z;
            qFile3 << "\n";
            qFile3 << threadParams->pendulum3.rotation.w;
            qFile3 << "\n";

            pFile3 << threadParams->pendulum3.position[eulerX];
            pFile3 << "\n";
            pFile3 << threadParams->pendulum3.position[eulerY];
            pFile3 << "\n";
            pFile3 << threadParams->pendulum3.position[eulerZ];
            pFile3 << "\n";
        }

        // Checking if stick is inside bounds
        if((threadParams->pendulum.massVector.rotate(threadParams->pendulum.rotation).y < 0) ||
            (threadParams->pendulum.position[eulerX] < -BOX_SIDES) ||
            (threadParams->pendulum.position[eulerX] > BOX_SIDES) ||
            (threadParams->pendulum.position[eulerZ] < -BOX_SIDES) ||
            (threadParams->pendulum.position[eulerZ] > BOX_SIDES)) {
            // stop the simulation, stick is out of bounds
            break;
        }
        if((threadParams->pendulum2.massVector.rotate(threadParams->pendulum2.rotation).y < 0) ||
            (threadParams->pendulum2.position[eulerX] < -BOX_SIDES) ||
            (threadParams->pendulum2.position[eulerX] > BOX_SIDES) ||
            (threadParams->pendulum2.position[eulerZ] < -BOX_SIDES) ||
            (threadParams->pendulum2.position[eulerZ] > BOX_SIDES)) {
            // stop the simulation, stick is out of bounds
            break;
        }
        if((threadParams->pendulum3.massVector.rotate(threadParams->pendulum3.rotation).y < 0) ||
            (threadParams->pendulum3.position[eulerX] < -BOX_SIDES) ||
            (threadParams->pendulum3.position[eulerX] > BOX_SIDES) ||
            (threadParams->pendulum3.position[eulerZ] < -BOX_SIDES) ||
            (threadParams->pendulum3.position[eulerZ] > BOX_SIDES)) {
            // stop the simulation, stick is out of bounds
            break;
        }

        (threadParams->result)++;
    }

    // if the run was recorded
    if(threadParams->should_save_best) {
        qFile.close();
        pFile.close();
        qFile2.close();
        pFile2.close();
        qFile3.close();
        pFile3.close();
    }
}








void renderGUI(GUI_param *param) {
    bool quitGUI = false;
    float worstInGenerations[MAX_GENERATIONS];
    float bestInGenerations[MAX_GENERATIONS];
    float generationsAverage[MAX_GENERATIONS];
    unsigned int lastGeneration = 0;
    float sessionBest = 0;
    ERR_E errLocal = ERR_OK;

    // Create application window
    ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, _T("ImGui Example"), NULL };
    ::RegisterClassEx(&wc);
    HWND hwnd = ::CreateWindow(wc.lpszClassName, _T("Stick balancing"), WS_OVERLAPPEDWINDOW, 100, 100, 2560, 1440, NULL, NULL, wc.hInstance, NULL);

    // Initialize Direct3D
    if(!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
    }

    // Show the window
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
                                                                //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows

    ImGui::StyleColorsDark();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    while(!quitGUI) {
        MSG msg;
        // Inputs from gui
        while(::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if(msg.message == WM_QUIT) {
                quitGUI = true;
                param->stop = true;
                thread_pool_stop(&errLocal);
            }
        }
        if(quitGUI) {
            break;
        }

        if(param->newGenerationDone) {
            lastGeneration = param->generation;
            worstInGenerations[lastGeneration] = param->worstInGeneration;
            bestInGenerations[lastGeneration]  = param->bestInGeneration;
            if(bestInGenerations[lastGeneration] > sessionBest) {
                sessionBest = bestInGenerations[lastGeneration];
            }
            generationsAverage[lastGeneration] = param->generationAverage;
            param->newGenerationDone = false;
        }

        // Start the Dear ImGui frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Plots");
        if(ImGui::Button("Stop")) {
            param->stop = true;
        }
#ifdef GPU_USE
        if(ImGui::Button("Save next GPU generation")) {
            param->saveGpuGen = true;
        }
#else
        if(ImGui::Button("Save next CPU generation")) {
            param->saveCpuGen = true;
        }
#endif // #ifdef GPU_USE
        char overlay[100];
        ImGui::Text("Generation %d", lastGeneration);
        sprintf_s(overlay, "Last generation %f\tBest in session %f", bestInGenerations[lastGeneration], sessionBest);
        ImGui::PlotLines("Best in generation", bestInGenerations, lastGeneration + 1, lastGeneration + 1, overlay, 0.0f, sessionBest, ImVec2(0.0f, 400.0f));
        sprintf_s(overlay, "Last generation average %f", generationsAverage[lastGeneration]);
        ImGui::PlotLines("Generation average", generationsAverage, lastGeneration + 1, lastGeneration + 1, overlay, 0.0f, sessionBest, ImVec2(0.0f, 400.0f));
        sprintf_s(overlay, "Worst in next generation %f", worstInGenerations[lastGeneration]);
        ImGui::PlotLines("Worst in generation", worstInGenerations, lastGeneration + 1, lastGeneration + 1, overlay, 0.0f, sessionBest, ImVec2(0.0f, 400.0f));
        ImGui::SliderInt("Mutation chance per 1000", &(param->mutationSlider), 1, 20);
        ImGui::End();

        // Rendering
        ImGui::Render();
        const float clear_color_with_alpha[4] = { clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        if(io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }

        g_pSwapChain->Present(1, 0); // Present with vsync
    }

    // Cleanup
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClass(wc.lpszClassName, wc.hInstance);
}

// Helper functions

bool CreateDeviceD3D(HWND hWnd) {
    // Setup swap chain
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    //createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };
    if(D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext) != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if(g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = NULL; }
    if(g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = NULL; }
    if(g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
}

void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget() {
    if(g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = NULL; }
}

#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0 // From Windows SDK 8.1+ headers
#endif

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if(ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch(msg) {
    case WM_SIZE:
        if(g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;
    case WM_SYSCOMMAND:
        if((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    case WM_DPICHANGED:
        if(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DpiEnableScaleViewports) {
            //const int dpi = HIWORD(wParam);
            //printf("WM_DPICHANGED to %d (%.0f%%)\n", dpi, (float)dpi / 96.0f * 100.0f);
            const RECT* suggested_rect = (RECT*)lParam;
            ::SetWindowPos(hWnd, NULL, suggested_rect->left, suggested_rect->top, suggested_rect->right - suggested_rect->left, suggested_rect->bottom - suggested_rect->top, SWP_NOZORDER | SWP_NOACTIVATE);
        }
        break;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}


static void tasks_done_handler() {
    tasksDone = true;
}