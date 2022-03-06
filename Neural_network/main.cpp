#include<stdlib.h>
#include<iostream>
#include<fstream>
#include"nn.h"
#include"quaternion.h"
#include"stick_balancing.h"
#define _USE_MATH_DEFINES
#include<math.h>
#include<thread> 
#include<ctime>
#include<string>

#define hidden_layers		3						//!< number of hidden layers in neural network
#define input_layer 		14						//!< number of input values for neural network
#define hidden1 			32						//!< number of nodes in 1st layer
#define hidden2 			50						//!< number of nodes in 2nd layer
#define hidden3 			8						//!< number of nodes in 3rd layer
#define output_layer		2						//!< number of output values
#define population 			256						//!< AI population size
#define mutation 			10						//!< chance per 1000 of random matrix field value 
#define next_gen 			10						//!< number of best cars that will be used to create next generation
#define roulette_size 		2000					//!< size of an array to bias the better cars for selection in new generation

#define PENDULUM_MASS		1						//!< Bottom pendulum mass in kg
#define PENDULUM2_MASS		1						//!< Middle pendulum mass in kg
#define PENDULUM3_MASS		1						//!< Top pendulum mass in kg
#define PENDULUM_HEIGHT		1						//!< Pendulum height in meters
#define PENDULUM_Y_POS		0.5						//!< Pendulum mass center height
#define PENDULUM_X_ANGLE	0.03					//!< Arbitrary angle around X axis
#define PENDULUM_Z_ANGLE	0.03					//!< Arbitrary angle around Z axis

#define TIME_STEP			0.005					//!< Simulation time step in seconds
#define DURATION			60						//!< Simulation duration in seconds
#define ITERATIONS			DURATION / TIME_STEP	//!< Number of iterations in simulation

#define FORCE_CLIPOFF		50						//!< Maximum force that can be applied to an object (times its weigth)
#define AI_FORCE			12						//!< Maximum horizontal force that AI can apply to an object (times its weigth)
#define SIM_TIMEOUT			ITERATIONS				//!< Maximum time that simulation will run

#define BOX_SIDES			10						//!< Box sides size for training environment

#define FIXED_POINT_TEST	0						//!< If set to 1, stick will freely rotate around a fixed point

#define NUM_OF_BODIES		2						//!< Number of bodies that the force applies to

#if population < next_gen
#error next generation can not be greater than population
#endif // #if population < next_gen

using namespace std;

void stick_thread(Neural_network *nn, Stick pendulum, Stick pendulum2, Stick pendulum3,  unsigned int *result, bool should_save_best);

void main() {
    unsigned int i, best_id[next_gen], roulette[roulette_size], tc, j, results[population];
    unsigned int rows, columns, current_best = 0, previous_best = 0, generation = 1, best_res = 0, repeat_res = 0;
    volatile int current_rank = 0;
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
    double pointVelocity[eulerCount];
    string saveFile;
    ERR_E  errLocal = ERR_OK;
    long int scoreSum = 0;
    float averageScore;

    char *unityRotation = "C:/repo/unity/inverted pendulum/Assets/angles.txt";
    char *unityPosition = "C:/repo/unity/inverted pendulum/Assets/position.txt";
    char *unityRotation2 = "C:/repo/unity/inverted pendulum/Assets/angles2.txt";
    char *unityPosition2 = "C:/repo/unity/inverted pendulum/Assets/position2.txt";
    char *unityRotation3 = "C:/repo/unity/inverted pendulum/Assets/angles3.txt";
    char *unityPosition3 = "C:/repo/unity/inverted pendulum/Assets/position3.txt";
    ofstream qFile, pFile, qFile2, pFile2, qFile3, pFile3;

    Neural_network *nn[population], *temp_nn[population], *best_nn;
    Neural_network *loaded_nn = new Neural_network(input_layer, hidden1, hidden2, output_layer);
    thread multi_thread[population];

    srand((int)time(NULL) + (int)clock());

    // create neural networks
    for(i = 0; i < population; i++) {
#if hidden_layers == 3
        nn[i] = new Neural_network(input_layer, hidden1, hidden2, hidden3, output_layer);
        temp_nn[i] = new Neural_network(input_layer, hidden1, hidden2, hidden3, output_layer);
        best_nn = new Neural_network(input_layer, hidden1, hidden2, hidden3, output_layer);
#elif hidden_layers == 2
        nn[i] = new Neural_network(input_layer, hidden1, hidden2, output_layer);
        temp_nn[i] = new Neural_network(input_layer, hidden1, hidden2, output_layer);
        best_nn = new Neural_network(input_layer, hidden1, hidden2, output_layer);
#elif hidden_layers == 1
        nn[i] = new Neural_network(input_layer, hidden1, output_layer);
        temp_nn[i] = new Neural_network(input_layer, hidden1, output_layer);
        best_nn = new Neural_network(input_layer, hidden1, output_layer);
#endif
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

#if FIXED_POINT_TEST == 0
    // run 1000 generations of the neural networks
    while(generation < 10000) {
        cout << "Generation " << generation << endl;

        // new random position of a stick for every generation
        pendulum.rotation.x = (double)((double)rand() / RAND_MAX * 2 * PENDULUM_X_ANGLE - PENDULUM_X_ANGLE);
        pendulum.rotation.z = (double)((double)rand() / RAND_MAX * 2 * PENDULUM_Z_ANGLE - PENDULUM_Z_ANGLE);
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

        // create threads
        for(tc = 0; tc < population; tc++) {
            multi_thread[tc] = thread(stick_thread, nn[tc], pendulum, pendulum2, pendulum3, &results[tc], false);
        }

        // wait for all threads to finnish
        for(tc = 0; tc < population; tc++) {
            multi_thread[tc].join();
        }

        // initialise the arrays
        for(i = 0; i < next_gen; i++) {
            best[i] = results[i];
            best_id[i] = i;
        }

        // sort the results
        for(i = 0; i < next_gen; i++) {
            for(tc = 0; tc < population; tc++) {
                if(results[tc] > best[i]) {
                    for(j = 0; j < i; j++) {
                        if(tc == best_id[j]) {
                            pass = true;
                        }
                    }
                    if(!pass) {
                        best[i] = results[tc];
                        best_id[i] = tc;
                    }
                    pass = false;
                }
            }
        }

        scoreSum = 0;

        for(i = 0; i < population; i++) {
            scoreSum += results[i];
        }

        averageScore = (double)scoreSum / (double)population;

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

        // check if it is the new session record
        if(results[best_id[0]] >= current_best) {

            cout << "--------------Best in session: " << results[best_id[0]];

            cout << "\tGeneration average: " << averageScore << "\tWorst in new generation: " << best[next_gen - 1] << endl;

            // set the new best score
            current_best = results[best_id[0]];

            // remember the best neural network
            best_nn->copy(*nn[best_id[0]]);
            best_res = results[best_id[0]];

            saveFile = "bis" + to_string(generation) + ".bin";

            nn[best_id[0]]->save(saveFile.c_str(), &errLocal);

            // repeat the run and record it this time
            stick_thread(best_nn, pendulum, pendulum2, pendulum3, &repeat_res, true);

        } else {
            cout << "Best in generation: " << results[best_id[0]];
            cout << "\t\t\tGeneration average: " << averageScore << "\tWorst in new generation: " << best[next_gen - 1] << endl;
        }

        // apply the new weights and add mutation
        if(results[best_id[0]] > previous_best) {

            // if it is new best, make one exact copy of the best neural network
            nn[0]->copy(*nn[best_id[0]]);

            // "randomize" the others
            for(i = 1; i < population; i++) {
                nn[i]->copy(*temp_nn[i]);
                nn[i]->randomize(mutation);
            }
            previous_best = current_best;
        } else {

            // "randomize" the next generation
            for(i = 0; i < population; i++) {
                nn[i]->copy(*(temp_nn[i]));
                nn[i]->randomize(mutation);
            }
        }
        generation++;
    } // while (generation < 1000)
#elif FIXED_POINT_TEST == 1
    // fixed point test

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
    remove(unityPosition3);
    pFile3.open(unityPosition3);

    for(i = 0; i < ITERATIONS; i++) {
        // calculate the force required to keep the base of the stick fixed
        pForce[eulerY] = -((pendulum.massVector.conjugate().rotate(pendulum.rotation)).y + pendulum.position[eulerY]) / TIME_STEP / TIME_STEP * pendulum.mass;
        pForce[eulerX] = -((pendulum.massVector.conjugate().rotate(pendulum.rotation)).x + pendulum.position[eulerX]) / TIME_STEP / TIME_STEP * pendulum.mass;
        pForce[eulerZ] = -((pendulum.massVector.conjugate().rotate(pendulum.rotation)).z + pendulum.position[eulerZ]) / TIME_STEP / TIME_STEP * pendulum.mass;

        // reduce the force if it is to big
        if(pForce[eulerY] < -pendulum.mass * g * FORCE_CLIPOFF) {
            pForce[eulerY] = -pendulum.mass * g * FORCE_CLIPOFF;
        }
        if(pForce[eulerY] > pendulum.mass * g * FORCE_CLIPOFF) {
            pForce[eulerY] = pendulum.mass * g * FORCE_CLIPOFF;
        }

        if(pForce[eulerX] > pendulum.mass * g * FORCE_CLIPOFF) {
            pForce[eulerX] = pendulum.mass * g * FORCE_CLIPOFF;
        }
        if(pForce[eulerX] < -pendulum.mass * g * FORCE_CLIPOFF) {
            pForce[eulerX] = -pendulum.mass * g * FORCE_CLIPOFF;
        }

        if(pForce[eulerZ] > pendulum.mass * g * FORCE_CLIPOFF) {
            pForce[eulerZ] = pendulum.mass * g * FORCE_CLIPOFF;
        }
        if(pForce[eulerZ] < -pendulum.mass * g * FORCE_CLIPOFF) {
            pForce[eulerZ] = -pendulum.mass * g * FORCE_CLIPOFF;
        }

        // joint force 1-2
        jointForce[eulerY] = (((pendulum.massVector.rotate(pendulum.rotation)).y + pendulum.position[eulerY]) -
            ((pendulum2.massVector.conjugate().rotate(pendulum2.rotation)).y + pendulum2.position[eulerY]))
            / TIME_STEP / TIME_STEP * pendulum.mass / NUM_OF_BODIES;
        jointForce[eulerX] = (((pendulum.massVector.rotate(pendulum.rotation)).x + pendulum.position[eulerX]) -
            ((pendulum2.massVector.conjugate().rotate(pendulum2.rotation)).x + pendulum2.position[eulerX]))
            / TIME_STEP / TIME_STEP * pendulum.mass / NUM_OF_BODIES;
        jointForce[eulerZ] = (((pendulum.massVector.rotate(pendulum.rotation)).z + pendulum.position[eulerZ]) -
            ((pendulum2.massVector.conjugate().rotate(pendulum2.rotation)).z + pendulum2.position[eulerZ]))
            / TIME_STEP / TIME_STEP * pendulum.mass / NUM_OF_BODIES;

        // reduce the force if it is to big
        if(jointForce[eulerY] < -pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerY] = -pendulum.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce[eulerY] > pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerY] = pendulum.mass * g * FORCE_CLIPOFF;
        }

        if(jointForce[eulerX] > pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerX] = pendulum.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce[eulerX] < -pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerX] = -pendulum.mass * g * FORCE_CLIPOFF;
        }

        if(jointForce[eulerZ] > pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerZ] = pendulum.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce[eulerZ] < -pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerZ] = -pendulum.mass * g * FORCE_CLIPOFF;
        }

        // joint force 3-4
        jointForce23[eulerY] = (((pendulum2.massVector.rotate(pendulum2.rotation)).y + pendulum2.position[eulerY]) -
            ((pendulum3.massVector.conjugate().rotate(pendulum3.rotation)).y + pendulum3.position[eulerY]))
            / TIME_STEP / TIME_STEP * pendulum2.mass / NUM_OF_BODIES;
        jointForce23[eulerX] = (((pendulum2.massVector.rotate(pendulum2.rotation)).x + pendulum2.position[eulerX]) -
            ((pendulum3.massVector.conjugate().rotate(pendulum3.rotation)).x + pendulum3.position[eulerX]))
            / TIME_STEP / TIME_STEP * pendulum2.mass / NUM_OF_BODIES;
        jointForce23[eulerZ] = (((pendulum2.massVector.rotate(pendulum2.rotation)).z + pendulum2.position[eulerZ]) -
            ((pendulum3.massVector.conjugate().rotate(pendulum3.rotation)).z + pendulum3.position[eulerZ]))
            / TIME_STEP / TIME_STEP * pendulum2.mass / NUM_OF_BODIES;

        // reduce the force if it is to big
        if(jointForce23[eulerY] < -pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerY] = -pendulum2.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce23[eulerY] > pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerY] = pendulum2.mass * g * FORCE_CLIPOFF;
        }

        if(jointForce23[eulerX] > pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerX] = pendulum2.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce23[eulerX] < -pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerX] = -pendulum2.mass * g * FORCE_CLIPOFF;
        }

        if(jointForce23[eulerZ] > pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerZ] = pendulum2.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce23[eulerZ] < -pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerZ] = -pendulum2.mass * g * FORCE_CLIPOFF;
        }

        // calculate the stick physics
        pendulum3.physics(jointForce23, zeroForce, TIME_STEP);

        // flip the force for pendulum2 because it has opposite reaction from top one
        jointForce23[eulerX] = -jointForce23[eulerX];
        jointForce23[eulerY] = -jointForce23[eulerY];
        jointForce23[eulerZ] = -jointForce23[eulerZ];

        pendulum2.physics(jointForce, jointForce23, TIME_STEP);

        // flip the force for bottom pendulum because it has opposite reaction from top one
        jointForce[eulerX] = -jointForce[eulerX];
        jointForce[eulerY] = -jointForce[eulerY];
        jointForce[eulerZ] = -jointForce[eulerZ];

        pendulum.physics(pForce, jointForce, TIME_STEP);

        // Write the bottom pendulum parameters to file
        qFile << pendulum.rotation.x;
        qFile << "\n";
        qFile << pendulum.rotation.y;
        qFile << "\n";
        qFile << pendulum.rotation.z;
        qFile << "\n";
        qFile << pendulum.rotation.w;
        qFile << "\n";

        pFile << pendulum.position[eulerX];
        pFile << "\n";
        pFile << pendulum.position[eulerY];
        pFile << "\n";
        pFile << pendulum.position[eulerZ];
        pFile << "\n";

        // Write the middle pendulum parameters to file
        qFile2 << pendulum2.rotation.x;
        qFile2 << "\n";
        qFile2 << pendulum2.rotation.y;
        qFile2 << "\n";
        qFile2 << pendulum2.rotation.z;
        qFile2 << "\n";
        qFile2 << pendulum2.rotation.w;
        qFile2 << "\n";

        pFile2 << pendulum2.position[eulerX];
        pFile2 << "\n";
        pFile2 << pendulum2.position[eulerY];
        pFile2 << "\n";
        pFile2 << pendulum2.position[eulerZ];
        pFile2 << "\n";

        // Write the top pendulum parameters to file
        qFile3 << pendulum3.rotation.x;
        qFile3 << "\n";
        qFile3 << pendulum3.rotation.y;
        qFile3 << "\n";
        qFile3 << pendulum3.rotation.z;
        qFile3 << "\n";
        qFile3 << pendulum3.rotation.w;
        qFile3 << "\n";

        pFile3 << pendulum3.position[eulerX];
        pFile3 << "\n";
        pFile3 << pendulum3.position[eulerY];
        pFile3 << "\n";
        pFile3 << pendulum3.position[eulerZ];
        pFile3 << "\n";

    }
#endif // #elif FIXED_POINT_TEST == 1

    cout << "Done" << endl;

    // keep the window opened
    cin >> generation;
}

void stick_thread(Neural_network *_nn, Stick pendulum, Stick pendulum2, Stick pendulum3, unsigned int *result, bool should_save_best) {
    Matrix inputs(input_layer, 1, false), outputs(output_layer, 1, false);
    double pendulum_height = pendulum.massVector.rotate(pendulum.rotation).y;
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

    // initialize the result
    *result = 0;

    // if the run is being recorded
    if(should_save_best) {
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

    // run until the simulation timeouts
    while(*result < SIM_TIMEOUT) {
        // calculate the stick euler angle to feed it as input
        pendulum.rotation.quaternionToEuler(eulerRotation);
        inputs.setel(0, 0, eulerRotation[eulerX]);
        inputs.setel(1, 0, eulerRotation[eulerZ]);
        // angular velocity
        inputs.setel(2, 0, pendulum.angularV[eulerX]);
        inputs.setel(3, 0, pendulum.angularV[eulerZ]);
        // position
        inputs.setel(4, 0, pendulum.position[eulerX]);
        inputs.setel(5, 0, pendulum.position[eulerZ]);

        pendulum2.rotation.quaternionToEuler(eulerRotation);
        inputs.setel(6, 0, eulerRotation[eulerX]);
        inputs.setel(7, 0, eulerRotation[eulerZ]);

        inputs.setel(8, 0, pendulum2.angularV[eulerX]);
        inputs.setel(9, 0, pendulum2.angularV[eulerZ]);

        pendulum3.rotation.quaternionToEuler(eulerRotation);
        inputs.setel(10, 0, eulerRotation[eulerX]);
        inputs.setel(11, 0, eulerRotation[eulerZ]);

        inputs.setel(12, 0, pendulum3.angularV[eulerX]);
        inputs.setel(13, 0, pendulum3.angularV[eulerZ]);

        // calculate the neural network output
        outputs.setmat(_nn->calculate(inputs));

        // set the horizontal forces from network output and calculate the vertical force to support the stick
        pForce[eulerX] = (outputs.getel(0, 0) - 0.5) * pendulum.mass * g * AI_FORCE * 2;
        pForce[eulerZ] = (outputs.getel(1, 0) - 0.5) * pendulum.mass * g * AI_FORCE * 2;
        pForce[eulerY] = -((pendulum.massVector.conjugate().rotate(pendulum.rotation)).y + pendulum.position[eulerY]) / TIME_STEP / TIME_STEP * pendulum.mass;
        // reduce the vertical force if it is to big
        if(pForce[eulerY] < -pendulum.mass * g * FORCE_CLIPOFF) {
            pForce[eulerY] = -pendulum.mass * g * FORCE_CLIPOFF;
        }
        if(pForce[eulerY] > pendulum.mass * g * FORCE_CLIPOFF) {
            pForce[eulerY] = pendulum.mass * g * FORCE_CLIPOFF;
        }

        // joint force 1-2
        jointForce[eulerY] = (((pendulum.massVector.rotate(pendulum.rotation)).y + pendulum.position[eulerY]) -
            ((pendulum2.massVector.conjugate().rotate(pendulum2.rotation)).y + pendulum2.position[eulerY]))
            / TIME_STEP / TIME_STEP * pendulum.mass / NUM_OF_BODIES;
        jointForce[eulerX] = (((pendulum.massVector.rotate(pendulum.rotation)).x + pendulum.position[eulerX]) -
            ((pendulum2.massVector.conjugate().rotate(pendulum2.rotation)).x + pendulum2.position[eulerX]))
            / TIME_STEP / TIME_STEP * pendulum.mass / NUM_OF_BODIES;
        jointForce[eulerZ] = (((pendulum.massVector.rotate(pendulum.rotation)).z + pendulum.position[eulerZ]) -
            ((pendulum2.massVector.conjugate().rotate(pendulum2.rotation)).z + pendulum2.position[eulerZ]))
            / TIME_STEP / TIME_STEP * pendulum.mass / NUM_OF_BODIES;

        // reduce the force if it is to big
        if(jointForce[eulerY] < -pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerY] = -pendulum.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce[eulerY] > pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerY] = pendulum.mass * g * FORCE_CLIPOFF;
        }

        if(jointForce[eulerX] > pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerX] = pendulum.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce[eulerX] < -pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerX] = -pendulum.mass * g * FORCE_CLIPOFF;
        }

        if(jointForce[eulerZ] > pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerZ] = pendulum.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce[eulerZ] < -pendulum.mass * g * FORCE_CLIPOFF) {
            jointForce[eulerZ] = -pendulum.mass * g * FORCE_CLIPOFF;
        }

        // joint force 3-4
        jointForce23[eulerY] = (((pendulum2.massVector.rotate(pendulum2.rotation)).y + pendulum2.position[eulerY]) -
            ((pendulum3.massVector.conjugate().rotate(pendulum3.rotation)).y + pendulum3.position[eulerY]))
            / TIME_STEP / TIME_STEP * pendulum2.mass / NUM_OF_BODIES;
        jointForce23[eulerX] = (((pendulum2.massVector.rotate(pendulum2.rotation)).x + pendulum2.position[eulerX]) -
            ((pendulum3.massVector.conjugate().rotate(pendulum3.rotation)).x + pendulum3.position[eulerX]))
            / TIME_STEP / TIME_STEP * pendulum2.mass / NUM_OF_BODIES;
        jointForce23[eulerZ] = (((pendulum2.massVector.rotate(pendulum2.rotation)).z + pendulum2.position[eulerZ]) -
            ((pendulum3.massVector.conjugate().rotate(pendulum3.rotation)).z + pendulum3.position[eulerZ]))
            / TIME_STEP / TIME_STEP * pendulum2.mass / NUM_OF_BODIES;

        // reduce the force if it is to big
        if(jointForce23[eulerY] < -pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerY] = -pendulum2.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce23[eulerY] > pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerY] = pendulum2.mass * g * FORCE_CLIPOFF;
        }

        if(jointForce23[eulerX] > pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerX] = pendulum2.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce23[eulerX] < -pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerX] = -pendulum2.mass * g * FORCE_CLIPOFF;
        }

        if(jointForce23[eulerZ] > pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerZ] = pendulum2.mass * g * FORCE_CLIPOFF;
        }
        if(jointForce23[eulerZ] < -pendulum2.mass * g * FORCE_CLIPOFF) {
            jointForce23[eulerZ] = -pendulum2.mass * g * FORCE_CLIPOFF;
        }

        // calculate the stick physics
        pendulum3.physics(jointForce23, zeroForce, TIME_STEP);

        // flip the force for pendulum2 because it has opposite reaction from top one
        jointForce23[eulerX] = -jointForce23[eulerX];
        jointForce23[eulerY] = -jointForce23[eulerY];
        jointForce23[eulerZ] = -jointForce23[eulerZ];

        pendulum2.physics(jointForce, jointForce23, TIME_STEP);

        // flip the force for bottom pendulum because it has opposite reaction from top one
        jointForce[eulerX] = -jointForce[eulerX];
        jointForce[eulerY] = -jointForce[eulerY];
        jointForce[eulerZ] = -jointForce[eulerZ];

        pendulum.physics(pForce, jointForce, TIME_STEP);

        // if the run is being recorded
        if(should_save_best) {
            // Write the stick parameters to file
            qFile << pendulum.rotation.x;
            qFile << "\n";
            qFile << pendulum.rotation.y;
            qFile << "\n";
            qFile << pendulum.rotation.z;
            qFile << "\n";
            qFile << pendulum.rotation.w;
            qFile << "\n";

            pFile << pendulum.position[eulerX];
            pFile << "\n";
            pFile << pendulum.position[eulerY];
            pFile << "\n";
            pFile << pendulum.position[eulerZ];
            pFile << "\n";

            // Write the middle pendulum parameters to file
            qFile2 << pendulum2.rotation.x;
            qFile2 << "\n";
            qFile2 << pendulum2.rotation.y;
            qFile2 << "\n";
            qFile2 << pendulum2.rotation.z;
            qFile2 << "\n";
            qFile2 << pendulum2.rotation.w;
            qFile2 << "\n";

            pFile2 << pendulum2.position[eulerX];
            pFile2 << "\n";
            pFile2 << pendulum2.position[eulerY];
            pFile2 << "\n";
            pFile2 << pendulum2.position[eulerZ];
            pFile2 << "\n";

            // Write the top pendulum parameters to file
            qFile3 << pendulum3.rotation.x;
            qFile3 << "\n";
            qFile3 << pendulum3.rotation.y;
            qFile3 << "\n";
            qFile3 << pendulum3.rotation.z;
            qFile3 << "\n";
            qFile3 << pendulum3.rotation.w;
            qFile3 << "\n";

            pFile3 << pendulum3.position[eulerX];
            pFile3 << "\n";
            pFile3 << pendulum3.position[eulerY];
            pFile3 << "\n";
            pFile3 << pendulum3.position[eulerZ];
            pFile3 << "\n";
        }

        // Checking if stick is inside bounds
        if((pendulum.massVector.rotate(pendulum.rotation).y < 0) ||
            (pendulum.position[eulerX] < -BOX_SIDES) ||
            (pendulum.position[eulerX] > BOX_SIDES) ||
            (pendulum.position[eulerZ] < -BOX_SIDES) ||
            (pendulum.position[eulerZ] > BOX_SIDES)) {
            // stop the simulation, stick is out of bounds
            break;
        }
        if((pendulum2.massVector.rotate(pendulum2.rotation).y < 0) ||
            (pendulum2.position[eulerX] < -BOX_SIDES) ||
            (pendulum2.position[eulerX] > BOX_SIDES) ||
            (pendulum2.position[eulerZ] < -BOX_SIDES) ||
            (pendulum2.position[eulerZ] > BOX_SIDES)) {
            // stop the simulation, stick is out of bounds
            break;
        }
        if((pendulum3.massVector.rotate(pendulum3.rotation).y < 0) ||
            (pendulum3.position[eulerX] < -BOX_SIDES) ||
            (pendulum3.position[eulerX] > BOX_SIDES) ||
            (pendulum3.position[eulerZ] < -BOX_SIDES) ||
            (pendulum3.position[eulerZ] > BOX_SIDES)) {
            // stop the simulation, stick is out of bounds
            break;
        }

        (*result)++;
    }

    // if the run was recorded
    if(should_save_best) {
        qFile.close();
        pFile.close();
        qFile2.close();
        pFile2.close();
        qFile3.close();
        pFile3.close();
    }
}