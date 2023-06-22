#include "CarregandoModelo.h"
#include <iostream>
#include <string>
#include <torch/script.h>
#include <filesystem>

using namespace std;

int main() {
	/*Já deixando a parte de obter path arbitrária independente de quem está executando
	desde que não mude o nome das pastas*/

	filesystem::path path_atual = filesystem::current_path();
	cout << "Path atual: " << path_atual << endl;

	string path_atual_novo = path_atual.string();

	size_t pos = path_atual_novo.find("libtorch");

	string path_tcc;
	if (pos != string::npos) {
		path_tcc = path_atual_novo.substr(0, pos);
	}

	string path_modelo = path_tcc + "modelos\\teste.pt";

	cout << "Path modelo: " << path_modelo << endl;

	if (filesystem::exists(path_modelo)) {
		cout << "Arquivo encontrado" << endl;
	}
	else {
		cout << "Arquivo não encontrado" << endl;
	}

	// Carregando o Modelo
	torch::jit::script::Module teste = torch::jit::load(path_modelo);

	// Criando tensor input
	torch::Tensor x = torch::tensor({ {142.0} });
	cout << "Input Tensor: " << x << endl;

	// Fazendo a inferência
	vector <torch::jit::IValue> input;
	input.push_back(x);
	auto out = teste.forward(input);
	cout << "Output Tensor: " << out << endl;

	// Lapidando resultado
	float normalizar_enth = -89587.13674;
	float enth_normalizada = out.toTensor().item<float>() + normalizar_enth;
	cout << "Resultado Entalpia: " << enth_normalizada << endl;

	return 0;
}