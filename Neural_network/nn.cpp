#include"nn.h"
#include<fstream>

#define MAX_NN_LAYERS           3
#define MAX_INPUTS              32
#define MAX_HIDDEN_LAYER_1      2048
#define MAX_HIDDEN_LAYER_2      2048
#define MAX_HIDDEN_LAYER_3      2048
#define MAX_OUTPUTS             32

using namespace std;

Neural_network::Neural_network() : layers(0), il(0), hl1(0), hl2(0), hl3(0), ol(0) {}

Neural_network::Neural_network(int _il, int _hl1, int _hl2, int _ol)
    : layers(2), il(_il), hl1(_hl1), hl2(_hl2), hl3(0), ol(_ol) {

    ih1 = new Matrix(hl1, il, true);
    h12 = new Matrix(hl2, hl1, true);
    ho = new Matrix(ol, hl2, true);

    r1 = new Matrix(hl1, 1, false);
    r2 = new Matrix(hl2, 1, false);

    b1 = new Matrix(hl1, 1, true);
    b2 = new Matrix(hl2, 1, true);
    ob = new Matrix(ol, 1, true);
    outputs = new Matrix(ol, 1, false);
}

Neural_network::Neural_network(int _il, int _hl, int _ol)
    : layers(1), il(_il), hl1(_hl), hl2(0), hl3(0), ol(_ol) {

    ih1 = new Matrix(hl1, il, true);
    ho = new Matrix(ol, hl1, true);

    r1 = new Matrix(hl1, 1, false);

    b1 = new Matrix(hl1, 1, true);
    ob = new Matrix(ol, 1, true);
    outputs = new Matrix(ol, 1, false);
}

Neural_network::Neural_network(int _il, int _hl1, int _hl2, int _hl3, int _ol)
    : layers(3), il(_il), hl1(_hl1), hl2(_hl2), hl3(_hl3), ol(_ol) {

    ih1 = new Matrix(hl1, il, true);
    h12 = new Matrix(hl2, hl1, true);
    h23 = new Matrix(hl3, hl2, true);
    ho = new Matrix(ol, hl3, true);

    r1 = new Matrix(hl1, 1, false);
    r2 = new Matrix(hl2, 1, false);
    r3 = new Matrix(hl3, 1, false);

    b1 = new Matrix(hl1, 1, true);
    b2 = new Matrix(hl2, 1, true);
    b3 = new Matrix(hl3, 1, true);
    ob = new Matrix(ol, 1, true);
    outputs = new Matrix(ol, 1, false);
}

Neural_network::Neural_network(const char *fileName) {
    FILE *saveFile;

    fopen_s(&saveFile, fileName, "rb");

    if(saveFile) {
        fread(&layers, sizeof(int), 1, saveFile);
        fread(&il, sizeof(int), 1, saveFile);
        fread(&hl1, sizeof(int), 1, saveFile);
        fread(&hl2, sizeof(int), 1, saveFile);
        fread(&hl3, sizeof(int), 1, saveFile);
        fread(&ol, sizeof(int), 1, saveFile);

        if((layers <= MAX_NN_LAYERS) &&
            (il <= MAX_INPUTS) &&
            (hl1 <= MAX_HIDDEN_LAYER_1) &&
            (hl2 <= MAX_HIDDEN_LAYER_2) &&
            (hl3 <= MAX_HIDDEN_LAYER_3) &&
            (ol <= MAX_OUTPUTS)) {

            if(layers == 1) {

                ih1 = new Matrix(hl1, il, true);
                r1 = new Matrix(hl1, 1, false);
                b1 = new Matrix(hl1, 1, true);
                ob = new Matrix(ol, 1, true);
                ho = new Matrix(ol, hl1, true);
                outputs = new Matrix(ol, 1, false);

            } else if(layers == 2) {

                ih1 = new Matrix(hl1, il, true);
                h12 = new Matrix(hl2, hl1, true);
                ho = new Matrix(ol, hl2, true);

                r1 = new Matrix(hl1, 1, false);
                r2 = new Matrix(hl2, 1, false);

                b1 = new Matrix(hl1, 1, true);
                b2 = new Matrix(hl2, 1, true);
                ob = new Matrix(ol, 1, true);
                outputs = new Matrix(ol, 1, false);

            } else if(layers == MAX_NN_LAYERS) {

                ih1 = new Matrix(hl1, il, true);
                h12 = new Matrix(hl2, hl1, true);
                h23 = new Matrix(hl3, hl2, true);
                ho = new Matrix(ol, hl3, true);

                r1 = new Matrix(hl1, 1, false);
                r2 = new Matrix(hl2, 1, false);
                r3 = new Matrix(hl3, 1, false);

                b1 = new Matrix(hl1, 1, true);
                b2 = new Matrix(hl2, 1, true);
                b3 = new Matrix(hl3, 1, true);
                ob = new Matrix(ol, 1, true);
                outputs = new Matrix(ol, 1, false);

            }

            fread(ih1->matrix, sizeof(double), ih1->row*ih1->col, saveFile);
            fread(r1->matrix, sizeof(double), r1->row*r1->col, saveFile);
            fread(b1->matrix, sizeof(double), b1->row*b1->col, saveFile);
            fread(ob->matrix, sizeof(double), ob->row*ob->col, saveFile);
            fread(ho->matrix, sizeof(double), ho->row*ho->col, saveFile);
            fread(outputs->matrix, sizeof(double), outputs->row*outputs->col, saveFile);

            if(layers > 1) {

                fread(h12->matrix, sizeof(double), h12->row*h12->col, saveFile);
                fread(r2->matrix, sizeof(double), r2->row*r2->col, saveFile);
                fread(b2->matrix, sizeof(double), b2->row*b2->col, saveFile);

                if(layers > 2) {

                    fread(h23->matrix, sizeof(double), h23->row*h23->col, saveFile);
                    fread(r3->matrix, sizeof(double), r3->row*r3->col, saveFile);
                    fread(b3->matrix, sizeof(double), b3->row*b3->col, saveFile);
                }
            }
        }

        fclose(saveFile);

    } else {
        layers = 0;
    }
}

Matrix Neural_network::calculate(Matrix inputs) {

    if(layers == 2) {

        mul(*ih1, inputs, r1);
        addi(r1, *b1);
        sigm(r1);
        mul(*h12, *r1, r2);
        addi(r2, *b2);
        sigm(r2);
        mul(*ho, *r2, outputs);
        addi(outputs, *ob);
        sigm(outputs);
        return *outputs;

    } else if(layers == 1) {

        mul(*ih1, inputs, r1);
        addi(r1, *b1);
        sigm(r1);
        mul(*ho, *r1, outputs);
        addi(outputs, *ob);
        sigm(outputs);
        return *outputs;

    } else if(layers == 3) {

        mul(*ih1, inputs, r1);
        addi(r1, *b1);
        sigm(r1);
        mul(*h12, *r1, r2);
        addi(r2, *b2);
        sigm(r2);
        mul(*h23, *r2, r3);
        addi(r3, *b3);
        sigm(r3);
        mul(*ho, *r3, outputs);
        addi(outputs, *ob);
        sigm(outputs);
        return *outputs;

    } else {
        return *outputs;
    }
}

void Neural_network::randomize(int mutation) {

    if(layers == 2) {

        ih1->randomize(mutation);
        h12->randomize(mutation);
        ho->randomize(mutation);
        b1->randomize(mutation);
        b2->randomize(mutation);
        ob->randomize(mutation);

    } else if(layers == 3) {

        ih1->randomize(mutation);
        h12->randomize(mutation);
        h23->randomize(mutation);
        ho->randomize(mutation);
        b1->randomize(mutation);
        b2->randomize(mutation);
        b3->randomize(mutation);
        ob->randomize(mutation);

    } else if(layers == 1) {

        ih1->randomize(mutation);
        ho->randomize(mutation);
        b1->randomize(mutation);
        ob->randomize(mutation);
    }
}

void Neural_network::clean() {

    if(layers == 2) {

        ih1->clean();
        h12->clean();
        ho->clean();
        b1->clean();
        b2->clean();
        ob->clean();

    } else if(layers == 3) {

        ih1->clean();
        h12->clean();
        h23->clean();
        ho->clean();
        b1->clean();
        b2->clean();
        b3->clean();
        ob->clean();

    } else if(layers == 1) {

        ih1->clean();
        ho->clean();
        b1->clean();
        ob->clean();
    }
}

void Neural_network::copy(Neural_network a) {

    if(layers == 2) {

        ih1->setmat(*(a.ih1));
        h12->setmat(*(a.h12));
        ho->setmat(*(a.ho));
        b1->setmat(*(a.b1));
        b2->setmat(*(a.b2));
        ob->setmat(*(a.ob));

    } else if(layers == 3) {

        ih1->setmat(*(a.ih1));
        h12->setmat(*(a.h12));
        h23->setmat(*(a.h23));
        ho->setmat(*(a.ho));
        b1->setmat(*(a.b1));
        b2->setmat(*(a.b2));
        b3->setmat(*(a.b3));
        ob->setmat(*(a.ob));

    } else if(layers == 1) {

        ih1->setmat(*(a.ih1));
        ho->setmat(*(a.ho));
        b1->setmat(*(a.b1));
        ob->setmat(*(a.ob));
    }
}

void Neural_network::save(const char *fileName, ERR_E *err) {
    FILE *saveFile;

    if(err != NULL) {
        remove(fileName);
        fopen_s(&saveFile, fileName, "wb");

        if(saveFile) {
            fwrite(&layers, sizeof(int), 1, saveFile);
            fwrite(&il, sizeof(int), 1, saveFile);
            fwrite(&hl1, sizeof(int), 1, saveFile);
            fwrite(&hl2, sizeof(int), 1, saveFile);
            fwrite(&hl3, sizeof(int), 1, saveFile);
            fwrite(&ol, sizeof(int), 1, saveFile);
            fwrite(ih1->matrix, sizeof(double), ih1->row*ih1->col, saveFile);
            fwrite(r1->matrix, sizeof(double), r1->row*r1->col, saveFile);
            fwrite(b1->matrix, sizeof(double), b1->row*b1->col, saveFile);
            fwrite(ob->matrix, sizeof(double), ob->row*ob->col, saveFile);
            fwrite(ho->matrix, sizeof(double), ho->row*ho->col, saveFile);
            fwrite(outputs->matrix, sizeof(double), outputs->row*outputs->col, saveFile);

            if(layers > 1) {
                fwrite(h12->matrix, sizeof(double), h12->row*h12->col, saveFile);
                fwrite(r2->matrix, sizeof(double), r2->row*r2->col, saveFile);
                fwrite(b2->matrix, sizeof(double), b2->row*b2->col, saveFile);

                if(layers > 2) {
                    fwrite(h23->matrix, sizeof(double), h23->row*h23->col, saveFile);
                    fwrite(r3->matrix, sizeof(double), r3->row*r3->col, saveFile);
                    fwrite(b3->matrix, sizeof(double), b3->row*b3->col, saveFile);
                }
            }

            fclose(saveFile);

        } else {
            *err = ERR_FILE_OPEN;
        }
    }
}

void Neural_network::load(const char *fileName, ERR_E *err) {
    int _layers, _il, _hl1, _hl2, _hl3, _ol;
    FILE *saveFile;

    if(err != NULL) {
        fopen_s(&saveFile, fileName, "rb");

        if(saveFile) {
            fread(&_layers, sizeof(int), 1, saveFile);
            fread(&_il, sizeof(int), 1, saveFile);
            fread(&_hl1, sizeof(int), 1, saveFile);
            fread(&_hl2, sizeof(int), 1, saveFile);
            fread(&_hl3, sizeof(int), 1, saveFile);
            fread(&_ol, sizeof(int), 1, saveFile);

            if((layers == _layers) &&
                (il == _il) &&
                (hl1 == _hl1) &&
                (hl2 == _hl2) &&
                (hl3 == _hl3) &&
                (ol == _ol)) {

                layers = _layers;
                il = _il;
                hl1 = _hl1;
                hl2 = _hl2;
                hl3 = _hl3;
                ol = _ol;

                fread(ih1->matrix, sizeof(double), ih1->row*ih1->col, saveFile);
                fread(r1->matrix, sizeof(double), r1->row*r1->col, saveFile);
                fread(b1->matrix, sizeof(double), b1->row*b1->col, saveFile);
                fread(ob->matrix, sizeof(double), ob->row*ob->col, saveFile);
                fread(ho->matrix, sizeof(double), ho->row*ho->col, saveFile);
                fread(outputs->matrix, sizeof(double), outputs->row*outputs->col, saveFile);

                if(layers > 1) {
                    fread(h12->matrix, sizeof(double), h12->row*h12->col, saveFile);
                    fread(r2->matrix, sizeof(double), r2->row*r2->col, saveFile);
                    fread(b2->matrix, sizeof(double), b2->row*b2->col, saveFile);

                    if(layers > 2) {
                        fread(h23->matrix, sizeof(double), h23->row*h23->col, saveFile);
                        fread(r3->matrix, sizeof(double), r3->row*r3->col, saveFile);
                        fread(b3->matrix, sizeof(double), b3->row*b3->col, saveFile);
                    }
                }
            } else {
                *err = ERR_INVALID_ARG;
            }

            fclose(saveFile);

        } else {
            *err = ERR_FILE_OPEN;
        }
    }
}