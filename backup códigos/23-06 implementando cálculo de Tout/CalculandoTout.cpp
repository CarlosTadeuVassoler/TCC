#include "CalculandoTout.h"
#include <iostream>
#include <string>
#include <torch/script.h>
#include <filesystem>
#include <map>

using namespace std;

string obter_path_modelo(string caso_estudo, string nome_modelo) {
	// obtém o path atual, bem como a posição da pasta TCC neste path
	string path_atual = filesystem::current_path().string();
	size_t pos = path_atual.find("TCC_bruto");

	// corta o path até a parte TCC/
	string path_tcc;
	if (pos != string::npos) {
		path_tcc = path_atual.substr(0, pos + 10);
	}

	// adiciona o arquivo modelos/nome_modelo no path TCC/
	string path_modelo = path_tcc + "modelos\\" + caso_estudo + "\\" + nome_modelo;

	return path_modelo;
}

map <string, vector<torch::jit::script::Module>> carregar_modelos(string caso_estudo,
																  vector<string> compostos,
																  map<string, vector<torch::jit::script::Module>> modelos)
{
	for (int i = 0; i < compostos.size(); i++) {
		// cria um vetor que em: 0 - modelo TxH; 1 - modelo HxT.
		vector<torch::jit::script::Module> vetor_modelos;

		// carrega o modelo e insere no vetor_modelos
		vetor_modelos.push_back(torch::jit::load(obter_path_modelo(caso_estudo, compostos[i] + "_TxH.pt")));
		vetor_modelos.push_back(torch::jit::load(obter_path_modelo(caso_estudo, compostos[i] + "_HxT.pt")));
		
		// insere os modelos carregados no mapa
		modelos[compostos[i]] = vetor_modelos;
	}

	return modelos;
}

float inferencia(float input,
				 torch::jit::script::Module modelo)
{
	// transforma o tensor do input em tensor, e em seguida em vetor para que o modelo possa utiliza-lo
	torch::Tensor input_tensor = torch::tensor({ {input} });
	vector <torch::jit::IValue> input_vector;
	input_vector.push_back(input_tensor);

	// faz a inferência e normaliza o output (entalpias foram transformadas em positivas pra não dar problema com relu)
	auto output = modelo.forward(input_vector);
	float resultado = output.toTensor().item<float>();

	return resultado;
}

float normalizar (string composto,
				  string qual_norma,
				  string direcao,
				  float input)
{
	// mapa que conté os dados para efetuar as normalizações e desnormalizações dos valores
	map<string, vector<float>> dados_normalizacao = {
		// {composto-qual_norma, {media, desvio, minimo}}
		{"n-butane-temp", {450, 28.89636655359978, -1.7303213505149568}},
		{"n-butane-enth", {-115.05496649280721, 7.89054021170493, -1.9393974679315724}},
		{"isopentane-temp", {450.0, 28.89636655359978, -1.7303213505149568}},
		{"isopentane-enth", {-144.46325840589412, 10.52085252229431, -1.454943747350219}},
		{"n-pentane-temp", {440.0, 28.89636655359978, -1.7303213505149568}},
		{"n-pentane-enth", {-142.92802986703296, 9.744838270011364, -1.3886041777224143}},
	};

	// obtendo os dados para a normalização desejada
	float media = dados_normalizacao[qual_norma][0];
	float desvio = dados_normalizacao[qual_norma][1];
	float minimo = dados_normalizacao[qual_norma][2];

	float valor_novo;

	if (direcao == "normal_to_modelo") {
		// se eu tenho um dado original e quero normalizar
		valor_novo = (input - media) / desvio - minimo;
	}
	else {
		// se eu tenho um dado normalizado e quero o original
		valor_novo = media + desvio * (input + minimo);
	}
}

float calcular_Tout(float Tin,
					float calor,
					torch::jit::script::Module modelo_TxH,
					torch::jit::script::Module modelo_HxT)
{
	float Tin_normalizada; //normalizar tin aqui
	float enth_atual_normalizada = inferencia(Tin_normalizada, modelo_HxT);
	float enth_atual; //desnormalizar enth_atual_normalizada aqui
	float enth_final = enth_atual + calor;
	float enth_final_normalizada; // normalizar enth_final aqui
	float temp_final_normalizada = inferencia(enth_final, modelo_TxH);
	float Tout; // desnormalizar temp_final_normalizada aqui
}


int main() {
	// nome da pasta com os modelos
	string caso_estudo = "caso-1-3-correntes";

	// lista com o nome dos compostos, que deve ser o começo do nome dos modelos
	vector <string> compostos = {
		"n-butane",
		"isopentane",
		"n-pentane"
	};

	// criando um mapa que contém todos os modelos de todas as correntes
	map <string, vector<torch::jit::script::Module>> modelos;
	modelos = carregar_modelos(caso_estudo, compostos, modelos);
	

	
	return 0;
}