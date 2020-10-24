#include"nn.h"

Neural_network::Neural_network()
	: layers(0), il(0), hl1(0), hl2(0), hl3(0), ol(0)
{

}

Neural_network::Neural_network(int _il, int _hl1, int _hl2, int _ol)
	: layers(2), il(_il), hl1(_hl1), hl2(_hl2), hl3(0), ol(_ol)
{
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
	: layers(1), il(_il), hl1(_hl), hl2(0), hl3(0), ol(_ol)
{
	ih1 = new Matrix(hl1, il, true);
	ho = new Matrix(ol, hl1, true);

	r1 = new Matrix(hl1, 1, false);

	b1 = new Matrix(hl1, 1, true);
	ob = new Matrix(ol, 1, true);
	outputs = new Matrix(ol, 1, false);
}

Neural_network::Neural_network(int _il, int _hl1, int _hl2, int _hl3, int _ol)
	: layers(3), il(_il), hl1(_hl1), hl2(_hl2), hl3(_hl3), ol(_ol)
{
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

Matrix Neural_network::calculate(Matrix inputs) {
	if (layers == 2) {
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
	}
	else if (layers == 1) {
		mul(*ih1, inputs, r1);
		addi(r1, *b1);
		sigm(r1);
		mul(*ho, *r1, outputs);
		addi(outputs, *ob);
		sigm(outputs);
		return *outputs;
	}
	else if (layers == 3) {
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
	}
	else {
		return *outputs;
	}
}

void Neural_network::randomize(int mutation) {
	if (layers == 2) {
		ih1->randomize(mutation);
		h12->randomize(mutation);
		ho->randomize(mutation);
		b1->randomize(mutation);
		b2->randomize(mutation);
		ob->randomize(mutation);
	}
	else if (layers == 3) {
		ih1->randomize(mutation);
		h12->randomize(mutation);
		h23->randomize(mutation);
		ho->randomize(mutation);
		b1->randomize(mutation);
		b2->randomize(mutation);
		b3->randomize(mutation);
		ob->randomize(mutation);
	}
	else if (layers == 1) {
		ih1->randomize(mutation);
		ho->randomize(mutation);
		b1->randomize(mutation);
		ob->randomize(mutation);
	}
}

void Neural_network::clean() {
	if (layers == 2) {
		ih1->clean();
		h12->clean();
		ho->clean();
		b1->clean();
		b2->clean();
		ob->clean();
	}
	else if (layers == 3) {
		ih1->clean();
		h12->clean();
		h23->clean();
		ho->clean();
		b1->clean();
		b2->clean();
		b3->clean();
		ob->clean();
	}
	else if (layers == 1) {
		ih1->clean();
		ho->clean();
		b1->clean();
		ob->clean();
	}
}

void Neural_network::copy(Neural_network a) {
	if (layers == 2) {
		ih1->setmat(*(a.ih1));
		h12->setmat(*(a.h12));
		ho->setmat(*(a.ho));
		b1->setmat(*(a.b1));
		b2->setmat(*(a.b2));
		ob->setmat(*(a.ob));
	}
	else if (layers == 3) {
		ih1->setmat(*(a.ih1));
		h12->setmat(*(a.h12));
		h23->setmat(*(a.h23));
		ho->setmat(*(a.ho));
		b1->setmat(*(a.b1));
		b2->setmat(*(a.b2));
		b3->setmat(*(a.b3));
		ob->setmat(*(a.ob));
	}
	else if (layers == 1) {
		ih1->setmat(*(a.ih1));
		ho->setmat(*(a.ho));
		b1->setmat(*(a.b1));
		ob->setmat(*(a.ob));
	}
}