#include "Heat-Int-IA.h"
#include <iostream> //header - iostream contains code for displaying things on screen
#include <string> //for manipulating/converting streams and numbers
#include <fstream> //for writing into files
#include <time.h> //for recording execution times
#include <random> //for generating random numbers from different distributions
#include <cmath> //for computing mathematical functions
#include <stdio.h> //printing on screen, file access, writing data to strings
#include <algorithm> //simple algorithms for manipulating matrices (sorting, max, min, etc)
#include <sstream>
#include <string>
#include <math.h>
#include <torch/script.h> // biblioteca para uso de redes neurais (Carlos)
#include <filesystem> // biblioteca para obter os paths de arquivos de maneira arbitrária


using namespace std;

struct CaseStudyStruct {
	//Aqui poder�o estar par�metros de algum problema em que se fa�a uso do operador Pinch
	//A e B sao apenas exemplos
	double A = 0;
	double B = 0;
};

struct PinchSolution {
	double dTmin;							//dTmin <<<vari�vel de decis�o>>>
	double VeldTmin;
	vector<double> HUFrac;					//Fra��o do requerimento energ�tico a ser usado em cada tipo de ut. quente <<<vari�vel de decis�o>>>
	vector<double> VelHUFrac;
	vector<double> CUFrac;					//Fra��o do requerimento energ�tico a ser usado em cada tipo de ut. fria <<<vari�vel de decis�o>>>
	vector<double> VelCUFrac;
	vector<double> VarHUTempOut;			//Tabela b�sica das utilidades quentes (sem piecewise, s� Tin Tout) <<<Coluna 3 (Tout) � vari�vel de decis�o>>>
	vector<double> VelVarHUTempOut;
	vector<double> VarCUTempOut;			//Tabela b�sica das utilidades frias (sem piecewise, s� Tin Tout) <<<Coluna 3 (Tout) � vari�vel de decis�o>>>
	vector<double> VelVarCUTempOut;
	double TotalOC;
	double TotalCC;
	double TAC;
};

struct SolutionStruct {
	//Aqui poder�o estar vari�veis de algum problema em que se fa�a uso do operador Pinch.
	//x e TotalCosts s�o apenas exemplos
	double x; //vari�vel hipot�tica
	double TotalCosts; //vari�vel de custo total
	double TPenCosts; //vari�vel de custo de penaliza��o
	PinchSolution PS; //Esta � a estrutura descrita na linha 14 (PinchSolution). Estruturas do tipo SolutionStruct possuem internamente uma estrutura do tipo PinchSolution
};

struct PinchInterm {
	vector<double> Areak;					//Superficie de troca termica de cada intervalo <<<calculado nesta fun��o>>>
	vector<double> HU;						//Total em energia usado para cada tipo de utilidade quente <<<calculado nesta fun��o>>>
	vector<double> CU;						//Total em energia usado para cada tipo de utilidade fria <<<calculado nesta fun��o>>>

	int intervals;							//No. de intervalos de entalpia <<<calculado nesta fun��o>>>

	vector<double> sumcpdtHU;				//Somat�rio de cp_i*dt_i para todos os pieces; realizado fora desta fun��o. Serve para calcular a vaz�o e; consequentemente CPz�o para todos os pieces da UQ <<<par�metro de entrada>>>
	vector<double> sumcpdtCU;				//Somat�rio de cp_i*dt_i para todos os pieces; realizado fora desta fun��o. Serve para calcular a vaz�o e; consequentemente CPz�o para todos os pieces da UF <<<par�metro de entrada>>>

	double TPenCosts;						//Custo de penaliza��o (se solu��o for inv�lida) <<<calculado nesta fun��o>>>
	vector<double> cascade1;				// -- As pr�ximas vari�veis s�o todas intermedi�rias para o c�lculo do Pinch e n�o ser�o descritas --
	vector<double> cascade2;
	vector<double> TableThin2;
	vector<double> TableThout2;
	vector<double> TableTcin2;
	vector<double> TableTcout2;
	vector<double> AllTemp;
	vector<double> AllTempFinal;
	vector<vector<double>> IntervalTable;
	vector<double> TableThin;
	vector<double> TableThout;
	vector<double> TableTcin;
	vector<double> TableTcout;
	vector<double> AllTh;
	vector<double> AllTc;
	vector<double> AllThFinal;
	vector<double> AllTcFinal;
	vector<double> AllThPW;
	vector<double> AllTcPW;
	vector<double> TableCPh;
	vector<double> TableCPc;
	vector<double> Tablehh;
	vector<double> Tablehc;
	vector<double> cascade1h;
	vector<double> cascade1c;
	int Nun;								//N�mero de unidades <<<Calculado nesta fun��o>>>

	vector<double> Tablephih;				//Par�metro calculado p/ corre��o em problemas com mais de um tipo de TC <<<calculado nesta fun��o>>>
	vector<double> Tablephic;				//Par�metro calculado p/ corre��o em problemas com mais de um tipo de TC <<<calculado nesta fun��o>>>
	vector<double> flowrateHU;				//Vaz�o de cada tipo de utilidade quente <<<calculado nesta fun��o>>>
	vector<double> flowrateCU;				//Vaz�o de cada tipo de utilidade fria <<<calculado nesta fun��o>>>

	vector<double> dT;						//--- Os pr�ximos par�metros s�o do m�todo Pinch e n�o ser�o detalhados ---
	vector<double> CPdiff;
	vector<double> dH;
	vector<int> IsCU;
	vector<int> IsHU;
	vector<double> dTh;
	vector<double> dHh;
	vector<double> CPdiffh;
	vector<double> dTc;
	vector<double> dHc;
	vector<double> CPdiffc;
	vector<double>  dTcPW;
	vector<double>  dThPW;
	vector<double>  CPdiffcPW;
	vector<double>  CPdiffhPW;
	vector<double>  dHcPW;
	vector<double>  dHhPW;
	vector<double>  cascade1cPW;
	vector<double> cascade1hPW;
	vector<vector<double>> qexch;
	vector<vector<double>> qexcc;
};

struct CaseStudyPinch {
	int noofstreams;						//No. original de correntes de processo <<<par�metro de entrada>>>
	int TotalStreams;						//No. total de pieces (soma de todos os pieces de todas as correntes de processo) <<<par�metro de entrada>>>
	int TotalHU;							//No. total de pieces de utilidades quentes <<<par�metro de entrada>>>
	int TotalCU;							//No. total de pieces de utilidades frias <<<par�metro de entrada>>>
	vector<double> Tin;						//Temperatura de entrada do piece <<<par�metro de entrada>>>
	vector<double> Tout;					//Temperatura de sa�da do piece <<<par�metro de entrada>>>
	vector<double> CP;						//CP (w*cp) do piece <<<par�metro de entrada>>>
	vector<double> h;						//Coef de transf de calor (constante, igual ao da corrente original em todos os pieces) <<<par�metro de entrada>>>
	vector<double> Thuin;					//Temperatura de entrada do piece (util quente) <<<par�metro de entrada>>>
	vector<double> Thuout;					//Temperatura de saida do piece (util quente) <<<par�metro de entrada>>>
	vector<double> CPhu;					//cpzinho de todos os pieces da util. quente <<<par�metro de entrada>>>
	vector<double> hhu;						//Coef de transf de calor da util quente (constante, igual ao da corrente original em todos os pieces) <<<par�metro de entrada>>>
	vector<double> Tcuin;					//Temperatura de entrada do piece (util fria) <<<par�metro de entrada>>>
	vector<double> Tcuout;					//Temperatura de saida do piece (util fria) <<<par�metro de entrada>>>
	vector<double> CPcu;					//cpzinho de todos os pieces da util. fria <<<par�metro de entrada>>>
	vector<double> hcu;						//Coef de transf de calor da util fria <<<par�metro de entrada>>>
	int TotalHUTemp;						//Total de tipos de utilidade quente <<<par�metro de entrada>>>
	int TotalCUTemp;						//Total de tipos de utilidade fria <<<par�metro de entrada>>>
	vector<double> CorrespNumHU;			//Numero da corrente original � qual cada ponto pertence
	vector<double> CorrespNumCU;			//Numero da corrente original � qual cada ponto pertence
	double B0;								//Custo fixo para unidade de refer�ncia <<<par�metro de entrada>>>
	double C0;								//Fator de custo de capital para unidade de refer�ncia <<<par�metro de entrada>>>
	double beta0;							//Expoente de CC para unidade de refer�ncia <<<par�metro de entrada>>>
	vector<double> B;						//Vetor com fatores de custo de capital individuais para TC correntes de processo (p/ cada piece) <<<par�metro de entrada>>>
	vector<double> C;						//Vetor com fatores de custo de capital individuais para TC correntes de processo (p/ cada piece) <<<par�metro de entrada>>>
	vector<double> beta;					//Vetor com beta individuais para TC correntes (p/ cada piece) <<<par�metro de entrada>>>
	vector<double> BHU;						//Vetor com fatores de custo de capital individuais para TC correntes de processo (p/ cada piece) <<<par�metro de entrada>>>
	vector<double> CHU;						//Fator de custo de capital para aquecedores (p/ cada piece) <<<par�metro de entrada>>>
	vector<double> betaHU;					//beta para aquecedores (p/ cada piece) <<<par�metro de entrada>>>
	vector<double> BCU;						//Vetor com fatores de custo de capital individuais para TC correntes de processo (p/ cada piece) <<<par�metro de entrada>>>
	vector<double> CCU;						//Fator de custo de capital para resfriadores (p/ cada piece) <<<par�metro de entrada>>>
	vector<double> betaCU;					//beta para resfriadores (p/ cada piece) <<<par�metro de entrada>>>
	vector<double> HUcosts;					//Custo de cada tipo de UQ <<<par�metro de entrada>>> 
	vector<double> CUcosts;					//Custo de cada tipo de UF <<<par�metro de entrada>>>
};

struct CaseStudyHENStruct {
	vector<vector<double>> Streams;
	vector<vector<double>> HUStreams;
	vector<vector<double>> CUStreams;
	vector<vector<double>> AllStreams;
	vector<vector<double>> AllHotStreams;
	vector<vector<double>> AllColdStreams;
	vector<double> HUCosts;
	vector<double> CUCosts;
	vector<double> B;
	vector<double> C;
	vector<double> beta;
	double B0;
	double C0;
	double beta0;
	vector<vector<double>> BB;
	vector<vector<double>> CC;
	vector<vector<double>> betaa;
	vector<double> Bh;
	vector<double> Ch;
	vector<double> betah;
	vector<double> Bc;
	vector<double> Cc;
	vector<double> betac;
	vector<double> AllHotB;
	vector<double> AllHotC;
	vector<double> AllHotbeta;
	vector<double> AllColdB;
	vector<double> AllColdC;
	vector<double> AllColdbeta;
	vector<double> freeendT;
	vector<double> ishotutil;
	vector<double> iscoldutil;
	int nhot;
	int ncold;
	int nhu;
	int ncu;

	int allnhot;
	int allncold;

	double AF;
	double hpery;

	vector<double> Qh;
	vector<double> Qhk;
	//vector<double> Qh;
	vector<double> hh;
	vector<double> CPh;
	vector<double> Thin;
	vector<double> Thfinal;

	vector<double> Qc;
	vector<double> Qck;
	//vector<double> Qc;
	vector<double> hc;
	vector<double> CPc;
	vector<double> Tcin;
	vector<double> Tcfinal;

	vector<double> hhu;
	vector<double> hcu;
	vector<double> Thuin;
	vector<double> Tcuin;
	vector<double> Thuout;
	vector<double> Tcuout;


	vector<vector<double>> U;
	vector<vector<double>> Uhu;
	vector<vector<double>> Ucu;

	double minarea = 0.0001;

	double taCtroc;
	double tbCtroc;

	double qaCtroc;
	double qbCtroc;

	double taCcu;
	double tbCcu;

	double taChu;
	double tbChu;

	double EMAT;

	double Qmin;
	double Frmin;

	int specialHE;
	char comment[8] = "";
};

struct HENSynInterm {
	vector<vector<double>> Thk;
	vector<vector<vector<double>>> Thout;
	vector<vector<double>> sumQThk;
	vector<double> Thfinal0;
	vector<vector<double>> Tck;
	vector<vector<vector<double>>> Tcout;
	vector<double> Tcfinal0;

	vector<vector<vector<double>>> LMTD;
	vector<vector<vector<double>>> Area;
	vector<double> Qhu;
	vector<double> Qcu;
	vector<double> LMTDhu;
	vector<double> LMTDcu;
	vector<double> Areahu;
	vector<double> Areacu;
	vector<double> TotalQcu;
	vector<double> TotalQhu;



	double TPenCosts;
	double AreaCosts;
	double UtilCosts;
};

struct HENSolutionStruct {
	vector<vector<vector<int>>> z;
	vector<vector<vector<double>>> Q;
	vector<vector<vector<double>>> Fh;
	vector<vector<vector<double>>> Fc;
	vector<vector<vector<double>>> VelQ;
	vector<vector<vector<double>>> VelFh;
	vector<vector<vector<double>>> VelFc;
	vector<vector<int>> zcu;
	vector<vector<int>> zhu;
	double TotalCosts = 0;
	double TPenCosts = 0;
	HENSynInterm HSI;
};

void Pinch(CaseStudyPinch CS, PinchSolution& PS, PinchInterm& PI, int print, double start, vector<int> today) {
	//	void ObjFun(int b[10][10], double x[10][10], double y[10][10], double Par1[10], double Par2[10], double Par3[10], double Par4[10], double Par5[10], double Par6[10], double& Cost1, double& Cost2, double& TotalCost) {

		//============== PINCH STARTS HERE ==============

		//for (s = 0; s <= TotalStreams - 1; s++) {
		//	MajorStreams[s][3] = constantA[s] + constantB[s] * Particle[pp].F2G;
		//}

	const int INTER = PI.Areak.size();

	double TotalCosts = 0;
	PI.TPenCosts = 0;

	double HUtotal = 0;
	double CUtotal = 0;
	double CompWTotal = 0;
	double ExpWTotal = 0;
	double HECC = 0;
	double WorkCC = 0;
	double TotalArea = 0;
	PI.Nun = 0;

	//int* array = new int[1000000];

	//double* PI.cascade1 = new double[INTER];
	/*
	double PI.cascade1[INTER] = {};
	double PI.cascade2[INTER] = {};

	double PI.TableThin2[INTER] = {};
	double PI.TableThout2[INTER] = {};
	double PI.TableTcin2[INTER] = {};
	double PI.TableTcout2[INTER] = {};

	double PI.AllTemp[INTER] = {};
	double PI.AllTempFinal[INTER] = {};

	double PI.IntervalTable[INTER][5] = {};

	double PI.TableThin[INTER] = {};
	double PI.TableThout[INTER] = {};
	double PI.TableTcin[INTER] = {};
	double PI.TableTcout[INTER] = {};

	double PI.AllTh[INTER] = {};
	double PI.AllTc[INTER] = {};
	double PI.AllThFinal[INTER] = {};
	double PI.AllTcFinal[INTER] = {};

	double PI.AllThPW[INTER] = {};
	double PI.AllTcPW[INTER] = {};

	double PI.TableCPh[INTER] = {};
	double PI.TableCPc[INTER] = {};
	double PI.Tablehh[INTER] = {};
	double PI.Tablehc[INTER] = {};


	vector<double> PI.cascade1; PI.cascade1.resize(INTER);
	vector<double> PI.cascade2; PI.cascade2.resize(INTER);

	vector<double>  PI.TableThin2; PI.TableThin2.resize(INTER);
	vector<double>  PI.TableThout2; PI.TableThout2.resize(INTER);
	vector<double>  PI.TableTcin2; PI.TableTcin2.resize(INTER);
	vector<double>  PI.TableTcout2; PI.TableTcout2.resize(INTER);

	vector<double>  PI.AllTemp; PI.AllTemp.resize(INTER);
	vector<double>  PI.AllTempFinal; PI.AllTempFinal.resize(INTER);

	vector<vector<double>> PI.IntervalTable; PI.IntervalTable.resize(INTER);
	for (int i = 0; i <= INTER - 1; i++) {
		PI.IntervalTable[i].resize(5);
	}

	vector<double>  PI.TableThin; PI.TableThin.resize(INTER);
	vector<double> PI.TableThout; PI.TableThout.resize(INTER);
	vector<double> PI.TableTcin; PI.TableTcin.resize(INTER);
	vector<double> PI.TableTcout; PI.TableTcout.resize(INTER);

	vector<double> PI.AllTh; PI.AllTh.resize(INTER);
	vector<double> PI.AllTc; PI.AllTc.resize(INTER);
	vector<double> PI.AllThFinal; PI.AllThFinal.resize(INTER);
	vector<double> PI.AllTcFinal; PI.AllTcFinal.resize(INTER);

	vector<double> PI.AllThPW; PI.AllThPW.resize(INTER);
	vector<double> PI.AllTcPW; PI.AllTcPW.resize(INTER);

	vector<double> PI.TableCPh; PI.TableCPh.resize(INTER);
	vector<double> PI.TableCPc; PI.TableCPc.resize(INTER);
	vector<double> PI.Tablehh; PI.Tablehh.resize(INTER);
	vector<double> PI.Tablehc; PI.Tablehc.resize(INTER);
	*/
	/*
	vector<double> PI.dT; PI.dT.resize(INTER);
	vector<double> PI.CPdiff; PI.CPdiff.resize(INTER);
	vector<double> PI.dH; PI.dH.resize(INTER);

	vector<int> PI.IsCU; PI.IsCU.resize(INTER);
	vector<int> PI.IsHU; PI.IsHU.resize(INTER);

	vector<double> PI.dTh; PI.dTh.resize(INTER);
	vector<double> PI.dHh; PI.dHh.resize(INTER);
	vector<double> PI.CPdiffh; PI.CPdiffh.resize(INTER);

	vector<double> PI.dTc; PI.dTc.resize(INTER);
	vector<double> PI.dHc; PI.dHc.resize(INTER);
	vector<double> PI.CPdiffc; PI.CPdiffc.resize(INTER);

	vector<double>  PI.dTcPW; PI.dTcPW.resize(INTER);
	vector<double>  PI.dThPW; PI.dThPW.resize(INTER);
	vector<double>  PI.CPdiffcPW; PI.CPdiffcPW.resize(INTER);
	vector<double>  PI.CPdiffhPW; PI.CPdiffhPW.resize(INTER);
	vector<double>  PI.dHcPW; PI.dHcPW.resize(INTER);
	vector<double>  PI.dHhPW; PI.dHhPW.resize(INTER);
	vector<double>  PI.cascade1cPW; PI.cascade1cPW.resize(INTER);
	vector<double> PI.cascade1hPW; PI.cascade1hPW.resize(INTER);

	vector<vector<double>> PI.qexch; PI.qexch.resize(INTER);
	vector<vector<double>> PI.qexcc; PI.qexcc.resize(INTER);
	vector<vector<int>> PI.PresentHotStreams; PI.PresentHotStreams.resize(INTER);
	vector<vector<int>> PI.PresentColdStreams; PI.PresentColdStreams.resize(INTER);

	for (int i = 0; i <= INTER - 1; i++) {
		//PI.PresentHotStreams[i].resize(INTER);
		//PI.PresentColdStreams[i].resize(INTER);
		PI.qexch[i].resize(INTER);
		PI.qexcc[i].resize(INTER);
	}

	*/
	vector<double> ThLM; ThLM.resize(INTER);
	vector<double> TcLM; TcLM.resize(INTER);




	//vector<double> qexc; qexc.resize(INTER);

	int s = 0;

	int conth = 0;
	int contallh = 0;
	int contc = 0;
	int contallc = 0;
	int contall = 0;
	int numstreams = 0;
	//double start = clock();
	for (s = 0; s <= CS.TotalStreams - 1; s++) {
		if (CS.Tin[s] > CS.Tout[s] && CS.CP[s] > 0.0) {
			numstreams++;

			PI.TableThin[conth] = CS.Tin[s] - PS.dTmin / 2;

			PI.TableThin2[conth] = CS.Tin[s];
			PI.AllTh[contallh] = CS.Tin[s];
			contallh++;

			PI.TableThout[conth] = CS.Tout[s] - PS.dTmin / 2;

			PI.TableThout2[conth] = CS.Tout[s];
			PI.AllTh[contallh] = CS.Tout[s];
			contallh++;
			PI.TableCPh[conth] = CS.CP[s];
			PI.Tablehh[conth] = CS.h[s];


			PI.AllTemp[contall] = PI.TableThin[conth];
			contall++;
			PI.AllTemp[contall] = PI.TableThout[conth];
			contall++;

			conth++;
		}
		else if (CS.Tin[s] < CS.Tout[s] && CS.CP[s] > 0.0) {
			numstreams++;
			PI.TableTcin[contc] = CS.Tin[s] + PS.dTmin / 2;

			PI.TableTcin2[contc] = CS.Tin[s];
			PI.AllTc[contallc] = CS.Tin[s];
			contallc++;

			PI.TableTcout[contc] = CS.Tout[s] + PS.dTmin / 2;

			PI.TableTcout2[contc] = CS.Tout[s];
			PI.AllTc[contallc] = CS.Tout[s];
			contallc++;
			PI.TableCPc[contc] = CS.CP[s];
			PI.Tablehc[contc] = CS.h[s];


			PI.AllTemp[contall] = PI.TableTcin[contc];
			contall++;
			PI.AllTemp[contall] = PI.TableTcout[contc];
			contall++;

			contc++;
		}

	}


	//grand composite
	int generalstreams = conth + contc;
	int nhot = conth;
	int ncold = contc;

	sort(PI.AllTemp.begin(), PI.AllTemp.begin() + contall, greater<double>());
	int cont2 = 0;
	int cont = 0;

	for (cont = 0; cont <= contall - 1; cont++) {
		if (abs(PI.AllTemp[cont] - PI.AllTemp[cont + 1]) >= 0) {
			PI.AllTempFinal[cont2] = PI.AllTemp[cont];
			cont2++;
		}
	}

	int numinter = cont2;



	int conti = 0;

	PI.dT[0] = 0;
	PI.CPdiff[0] = 0;
	PI.dH[0] = 0;
	PI.cascade1[0] = 0;

	double sumcph = 0;
	double sumcpc = 0;

	for (conti = 1; conti <= numinter - 1; conti++) {
		PI.dT[conti] = PI.AllTempFinal[conti - 1] - PI.AllTempFinal[conti];
		sumcph = 0;
		sumcpc = 0;
		for (conth = 0; conth <= nhot - 1; conth++) {
			if ((PI.TableThin[conth] - PI.AllTempFinal[conti] > 0) && (PI.TableThout[conth] - PI.AllTempFinal[conti] <= -0)) {
				sumcph = sumcph + PI.TableCPh[conth];
			}
		}
		for (contc = 0; contc <= ncold - 1; contc++) {
			if ((PI.TableTcout[contc] - PI.AllTempFinal[conti] > 0) && (PI.TableTcin[contc] - PI.AllTempFinal[conti] <= -0)) {
				sumcpc = sumcpc + PI.TableCPc[contc];
			}
		}
		PI.CPdiff[conti] = sumcpc - sumcph;
		PI.dH[conti] = PI.CPdiff[conti] * PI.dT[conti];
		PI.cascade1[conti] = PI.cascade1[conti - 1] - PI.dH[conti];
	}

	double maxcascade;
	double mincascade;

	//maxcascade = abs(*max_element(PI.cascade1, PI.cascade1 + numinter));
	mincascade = abs(*min_element(PI.cascade1.begin(), PI.cascade1.begin() + numinter));

	//if (maxcascade < mincascade) {
	//	maxcascade = mincascade;
	//}
	PI.cascade2[0] = mincascade;
	for (conti = 1; conti <= numinter - 1; conti++) {
		PI.cascade2[conti] = PI.cascade2[conti - 1] - PI.dH[conti];
	}

	int cascade12size = 0;

	cascade12size = numinter - 1;

	HUtotal = PI.cascade2[0];
	CUtotal = PI.cascade2[conti - 1];
	//CompWTotal = CompWTotal;
	//ExpWTotal = ExpWTotal;

	//Hot composite curve
	sort(PI.AllTh.begin(), PI.AllTh.begin() + contallh, greater<double>());
	cont2 = 0;
	for (cont = 0; cont <= contallh - 1; cont++) {
		if (abs(PI.AllTh[cont] - PI.AllTh[cont + 1]) >= 0) {
			PI.AllThFinal[cont2] = PI.AllTh[cont];
			PI.AllThPW[cont2 + 2 * CS.TotalHU] = PI.AllTh[cont];
			cont2++;
		}
	}


	int conthu = 0;
	int contcu = 0;

	for (conthu = 0; conthu <= CS.TotalHU - 1; conthu++) {
		PI.AllThPW[2 * conthu] = CS.Thuin[conthu];
		PI.IsHU[conthu] = 1;
		PI.AllThPW[2 * conthu + 1] = CS.Thuout[conthu];
		PI.IsHU[conthu] = 1;
	}

	/*
	PI.AllThPW[0] = CS.Thuin[0];
	PI.AllThPW[1] = CS.Thuout[0];
	PI.IsHU[0] = 0;
	PI.IsHU[1] = 1;
	*/

	numinter = cont2;



	//double PI.dT[INTER] = {};
	//double PI.CPdiff[INTER] = {};
	//double PI.dH[INTER] = {};
	//double PI.cascade1[INTER] = {};
	//double PI.cascade2[INTER] = {};

	if (numinter == 0) {
		int AAA = 0;
	}
	if (numinter > 0) {
		PI.dTh[numinter - 1] = 0;
		PI.CPdiffh[numinter - 1] = 0;
		PI.dHh[numinter - 1] = 0;
		PI.cascade1h[numinter - 1] = 0;
		//PI.cascade1hPW[numinter - 1 + 2] = 0;
		for (conti = numinter - 2; conti >= 0; conti--) {
			PI.dTh[conti] = PI.AllThFinal[conti] - PI.AllThFinal[conti + 1];
			sumcph = 0;
			sumcpc = 0;
			for (conth = 0; conth <= nhot - 1; conth++) {
				if ((PI.TableThin2[conth] - PI.AllThFinal[conti] >= 0) && (PI.TableThout2[conth] - PI.AllThFinal[conti] < -0)) {
					sumcph = sumcph + PI.TableCPh[conth];
				}
			}
			PI.CPdiffh[conti] = sumcph;
			//PI.CPdiffhPW[conti + 2] = sumcph;
			PI.dHh[conti] = PI.CPdiffh[conti] * PI.dTh[conti];
			PI.cascade1h[conti] = PI.cascade1h[conti + 1] + PI.dHh[conti];
			//PI.cascade1hPW[conti + 2] = PI.cascade1h[conti];
		}
	}



	int contallcPW = 0;
	int contallhPW = 0;
	int hotintervalsind = 0;
	int coldintervalsind = 0;
	//int cascade12size = 0;
	int cascade1hsize = 0;
	int cascade1csize = 0;

	int conttotal;
	int i = 1;
	int j = 1;
	double TempIntervalT[5] = {};
	int ii0;
	int iii;
	int ii1;
	int jj;
	int ii;
	vector<vector<int>> intervalstoreorder; intervalstoreorder.resize(2);
	for (int i = 0; i <= 2 - 1; i++) {
		intervalstoreorder[i].resize(INTER);
	}
	int TotalAreaIntervals = 0;
	double sumqhh = 0;
	double sumqhc = 0;
	double AreaPen = 0;
	double EMAT = 1;
	double taCtroc = 50000000 / 10;
	double tbCtroc = 50000000 / 10;
	vector<double> LMTDk; LMTDk.resize(INTER);// [INTER] ;
	//double PI.Areak[100];



	/*
	vector<vector<int>> zh; zh.resize((int)(INTER / 2));
	vector<vector<int>> zc; zc.resize((int)(INTER / 2));

	for (int i = 0; i <= (int)(INTER / 2) - 1; i++) {
		zh[i].resize(INTER);
		zc[i].resize(INTER);
	}

	vector<vector<vector<int>>> z0; z0.resize((int)(INTER / 2)); //[(int)(INTER / 2)][(int)(INTER / 2)][INTER] = {};
	vector<vector<vector<int>>> z; z.resize((int)(INTER / 2)); //[(int)(INTER / 2)][(int)(INTER / 2)][INTER] = {};

	for (int i = 0; i <= (int)(INTER / 2) - 1; i++) {
		z0[i].resize((int)(INTER / 2));
		z[i].resize((int)(INTER / 2));
		for (int j = 0; j <= (int)(INTER / 2) - 1; j++) {
			z0[i][j].resize(INTER);
			z[i][j].resize(INTER);
		}
	}
	*/




	int kfinal = 0;
	int kexists = 0;
	if (numinter > 0) {
		PI.dThPW[numinter - 1] = 0;
		PI.CPdiffhPW[numinter - 1] = 0;
		PI.dHhPW[numinter - 1] = 0;
		//PI.cascade1hPW[numinter - 1] = 0;
		PI.cascade1hPW[numinter - 1 + 2] = 0;
		contallhPW = contallh + 2 * CS.TotalHU;// contallh + 2;
		sort(PI.AllThPW.begin(), PI.AllThPW.begin() + contallhPW, greater<double>());
	}
	//vector<double> flowrateHU; 
	PI.flowrateHU.resize(CS.TotalHUTemp);
	//vector<double> flowrateCU; 
	PI.flowrateCU.resize(CS.TotalCUTemp);

	//HUTOTAL * HUFRAC[conthu] = W[conthu] * sum (piece, cpphu[conthu][piece] * (CS.Thuin[conthu][piece] - CS.Thuout[conthu][piece]));
	//vaz�o hot util
	//correcting heat capacity flowrates for utilities.
	//As utilidades dever�o ter os cps, e n�o os CPs.
	//Dever�o ent�o ser corrigidas. A vaz�o necess�ria � calculada e multiplicada pelo CS.CPhu ou CS.CPcu, que nesse momento s�o, na verdade, o cp (min�sculo). 
	//Portanto, nas tabelas CS.MajorHU e CS.MajorCU dever�o estar SEMPRE os valores de capacidade calor�fica (kW/kgK), e n�o capacit�ncia t�rmica (kW/K).
	vector<double> sumcpdtHU2; sumcpdtHU2.resize(CS.TotalHUTemp);
	for (conthu = 0; conthu <= CS.TotalHU - 1; conthu++) {
		sumcpdtHU2[(int)CS.CorrespNumHU[conthu] - 1] = sumcpdtHU2[(int)CS.CorrespNumHU[conthu] - 1] + CS.CPhu[conthu] * (CS.Thuin[conthu] - CS.Thuout[conthu]);
	}
	vector<double> sumcpdtCU2; sumcpdtCU2.resize(CS.TotalCUTemp);
	for (contcu = 0; contcu <= CS.TotalCU - 1; contcu++) {
		sumcpdtCU2[(int)CS.CorrespNumCU[contcu] - 1] = sumcpdtCU2[(int)CS.CorrespNumCU[contcu] - 1] + CS.CPcu[contcu] * (CS.Tcuout[contcu] - CS.Tcuin[contcu]);
	}

	for (conthu = 0; conthu <= CS.TotalHUTemp - 1; conthu++) {
		PI.sumcpdtHU[conthu] = sumcpdtHU2[conthu];
		PI.flowrateHU[conthu] = (PS.HUFrac[conthu] * HUtotal) / PI.sumcpdtHU[conthu];
	}
	for (contcu = 0; contcu <= CS.TotalCUTemp - 1; contcu++) {
		PI.sumcpdtCU[contcu] = sumcpdtCU2[contcu];
		PI.flowrateCU[contcu] = (PS.CUFrac[contcu] * CUtotal) / PI.sumcpdtCU[contcu];
	}

	for (conthu = 0; conthu <= CS.TotalHU - 1; conthu++) {
		CS.CPhu[conthu] = PI.flowrateHU[(int)CS.CorrespNumHU[conthu] - 1] * CS.CPhu[conthu];// (PS.HUFrac[conthu] * HUtotal) / (CS.Thuin[conthu] - CS.Thuout[conthu]);
	}
	for (contcu = 0; contcu <= CS.TotalCU - 1; contcu++) {
		CS.CPcu[contcu] = PI.flowrateCU[(int)CS.CorrespNumCU[contcu] - 1] * CS.CPcu[contcu];// (PS.CUFrac[contcu] * CUtotal) / (CS.Tcuout[contcu] - CS.Tcuin[contcu]);
	}

	vector<vector<int>> HUPresent; HUPresent.resize(INTER);
	vector<vector<int>> CUPresent; CUPresent.resize(INTER);
	for (int i = 0; i <= INTER - 1; i++) {
		HUPresent[i].resize(CS.TotalHU);
		CUPresent[i].resize(CS.TotalCU);
	}


	for (conti = (contallhPW - 1); conti >= 0; conti--) {
		PI.dThPW[conti] = PI.AllThPW[conti] - PI.AllThPW[conti + 1];
		sumcph = 0;
		sumcpc = 0;
		for (conth = 0; conth <= nhot - 1; conth++) {
			if ((PI.TableThin2[conth] - PI.AllThPW[conti] >= 0) && (PI.TableThout2[conth] - PI.AllThPW[conti] < -0)) {
				sumcph = sumcph + PI.TableCPh[conth];
			}
		}
		for (conthu = 0; conthu <= CS.TotalHU - 1; conthu++) {
			if ((CS.Thuin[conthu] - PI.AllThPW[conti] >= 0) && (CS.Thuout[conthu] - PI.AllThPW[conti] < -0)) {
				sumcph = sumcph + CS.CPhu[conthu]; // PS.HUFrac[conthu] * HUtotal / (CS.Thuin[conthu] - CS.Thuout[conthu]);//WRONG
				if (CS.CPhu[conthu] > 0) {
					HUPresent[conti][conthu] = 1;
				}
			}
		}
		PI.CPdiffhPW[conti] = sumcph;
		//PI.CPdiffhPW[conti + 2] = sumcph;
		PI.dHhPW[conti] = PI.CPdiffhPW[conti] * PI.dThPW[conti];
		PI.cascade1hPW[conti] = PI.cascade1hPW[conti + 1] + PI.dHhPW[conti];
		//PI.cascade1hPW[conti + 2] = PI.cascade1h[conti];
	}



	//PI.cascade1hPW[0] = (HUtotal + PI.cascade1h[0]);
	//PI.cascade1hPW[1] = PI.cascade1h[0];
	//PI.CPdiffhPW[0] = (PI.cascade1hPW[0] - PI.cascade1hPW[1]) / (PI.AllThPW[0] - PI.AllThPW[1]);
	//PI.CPdiffhPW[1] = 0;



	hotintervalsind = numinter - 1;

	cascade1hsize = numinter - 1;

	//Cold composite curve
	sort(PI.AllTc.begin(), PI.AllTc.begin() + contallc, greater<double>());
	cont2 = 0;
	for (cont = 0; cont <= contallc - 1; cont++) {
		if (abs(PI.AllTc[cont] - PI.AllTc[cont + 1]) >= 0) {
			PI.AllTcFinal[cont2] = PI.AllTc[cont];
			PI.AllTcPW[cont2] = PI.AllTcFinal[cont2];
			cont2++;
		}
	}
	for (contcu = 0; contcu <= CS.TotalCU - 1; contcu++) {
		PI.AllTcPW[2 * contcu + cont2] = CS.Tcuout[contcu];
		PI.IsCU[contcu] = 1;
		PI.AllTcPW[2 * contcu + cont2 + 1] = CS.Tcuin[contcu];
		PI.IsCU[contcu] = 1;
	}
	/*
	PI.AllTcPW[cont2] = CS.Tcuout[0];
	PI.AllTcPW[cont2 + 1] = CS.Tcuin[0];
	PI.IsCU[cont2] = 1;
	PI.IsCU[cont2 + 1] = 0;
	*/
	if (numinter == 0) {
		int AAA = 0;
	}
	numinter = cont2;
	if (numinter == 0) {
		int AAA = 0;
	}
	if (numinter > 0) {
		PI.dTc[numinter - 1] = 0;
		PI.CPdiffc[numinter - 1] = 0;
		PI.dHc[numinter - 1] = 0;
		PI.cascade1c[numinter - 1] = CUtotal;
		//PI.cascade1cPW[numinter - 1] = CUtotal;
		for (conti = numinter - 2; conti >= 0; conti--) {
			PI.dTc[conti] = PI.AllTcFinal[conti] - PI.AllTcFinal[conti + 1];
			sumcph = 0;
			sumcpc = 0;
			for (contc = 0; contc <= ncold - 1; contc++) {
				if ((PI.TableTcout2[contc] - PI.AllTcFinal[conti] >= 0) && (PI.TableTcin2[contc] - PI.AllTcFinal[conti] < -0)) {
					sumcpc = sumcpc + PI.TableCPc[contc];
				}
			}
			PI.CPdiffc[conti] = sumcpc;
			//PI.CPdiffcPW[conti] = sumcpc;
			PI.dHc[conti] = PI.CPdiffc[conti] * PI.dTc[conti];
			//PI.dHcPW[conti] = PI.dHc[conti];
			PI.cascade1c[conti] = PI.cascade1c[conti + 1] + PI.dHc[conti];
			//PI.cascade1cPW[conti] = PI.cascade1c[conti];
		}

		PI.dTcPW[numinter - 1] = 0;
		PI.CPdiffcPW[numinter - 1] = 0;
		PI.dHcPW[numinter - 1] = 0;
		PI.cascade1cPW[numinter - 1] = 0;
		contallcPW = contallc + 2 * CS.TotalCU;
		sort(PI.AllTcPW.begin(), PI.AllTcPW.begin() + contallcPW, greater<double>());
	}

	for (conti = (contallcPW - 1); conti >= 0; conti--) {
		PI.dTcPW[conti] = PI.AllTcPW[conti] - PI.AllTcPW[conti + 1];
		sumcph = 0;
		sumcpc = 0;
		for (contcu = 0; contcu <= CS.TotalCU - 1; contcu++) {
			if ((CS.Tcuout[contcu] - PI.AllTcPW[conti] >= 0) && (CS.Tcuin[contcu] - PI.AllTcPW[conti] < -0)) {
				sumcpc = sumcpc + CS.CPcu[contcu];//PS.CUFrac[contcu] * CUtotal / (CS.Tcuout[contcu] - CS.Tcuin[contcu]);
				if (CS.CPcu[contcu] > 0) {
					CUPresent[conti][contcu] = 1;
				}
			}
		}
		for (contc = 0; contc <= ncold - 1; contc++) {
			if ((PI.TableTcout2[contc] - PI.AllTcPW[conti] >= 0) && (PI.TableTcin2[contc] - PI.AllTcPW[conti] < -0)) {
				sumcpc = sumcpc + PI.TableCPc[contc];
			}
		}
		PI.CPdiffcPW[conti] = sumcpc;
		//PI.CPdiffcPW[conti] = sumcpc;
		PI.dHcPW[conti] = PI.CPdiffcPW[conti] * PI.dTcPW[conti];
		//PI.dHcPW[conti] = PI.dHc[conti];
		PI.cascade1cPW[conti] = PI.cascade1cPW[conti + 1] + PI.dHcPW[conti];
		if (PI.cascade1cPW[conti] < 0.000000001) {
			PI.cascade1cPW[conti] = 0;
		}
		//PI.cascade1cPW[conti] = PI.cascade1c[conti];
	}


	//PI.cascade1cPW[numinter] = PI.cascade1c[numinter - 1];
	//PI.cascade1cPW[numinter + 1] = 0.0;
	//PI.CPdiffcPW[numinter] = (PI.cascade1cPW[numinter + 1] - PI.cascade1cPW[numinter]) / (PI.AllTcPW[numinter + 1] - PI.AllTcPW[numinter]);

	coldintervalsind = numinter - 1;

	cascade1csize = numinter - 1;

	//Calc Thin/Thout
	int contii = 0;
	for (conti = 0; conti <= (contallcPW - 1); conti++) { //coldintervalsind + 2; conti++) {
		contc = 1;
		while ((PI.cascade1hPW[contc] - PI.cascade1cPW[conti]) > 0.00000000001) {//0.00000001) {
			contc++;
		}
		if (PI.CPdiffhPW[contc - 1] == 0) {
			ThLM[conti] = PI.AllThPW[contc];// +(PI.cascade1cPW[conti] - PI.cascade1hPW[contc]) / PI.CPdiffhPW[contc - 1];
		}
		else {
			ThLM[conti] = PI.AllThPW[contc] + (PI.cascade1cPW[conti] - PI.cascade1hPW[contc]) / PI.CPdiffhPW[contc - 1];
		}


		//if (Particle[0].PS.dTmin > 24.51 && Particle[0].PS.dTmin < 24.7) {
		//	printf("%.12f\t = %.12f\t - %.12f\t * %.12f\t\n", ThLM[conti], PI.AllThPW[contc] + (PI.cascade1cPW[conti]) / PI.CPdiffhPW[contc - 1], 1 / PI.CPdiffhPW[contc - 1], PI.cascade1hPW[contc]);
		//}
		//if (PI.CPdiffhPW[contc - 1] == 0) {
		//	AreaCosts = AreaCosts;
		//}

		//PI.IntervalTable[conti][0] = PI.cascade1cPW[conti];
		//PI.IntervalTable[conti][1] = ThLM[conti];
		//PI.IntervalTable[conti][2] = PI.AllTcPW[conti];
		//PI.IntervalTable[conti][4] = PI.IsCU[conti];

		//if (conti <= (contallcPW - 2)) {
		//	if (PI.cascade1cPW[conti] != PI.cascade1cPW[conti + 1]) {
		PI.IntervalTable[contii][0] = PI.cascade1cPW[conti];
		PI.IntervalTable[contii][1] = ThLM[conti];
		PI.IntervalTable[contii][2] = PI.AllTcPW[conti];
		PI.IntervalTable[contii][4] = PI.IsCU[conti];
		if (PI.IntervalTable[contii][2] > 333.99 && PI.IntervalTable[contii][2] < 334.01) {
			int aaaaaa = 1;
		}
		contii++;


		//	}
		//}
		/*
		else {
			PI.IntervalTable[contii][0] = PI.cascade1cPW[conti];
			PI.IntervalTable[contii][1] = ThLM[conti];
			PI.IntervalTable[contii][2] = PI.AllTcPW[conti];
			PI.IntervalTable[contii][4] = PI.IsCU[conti];
			if (PI.IntervalTable[contii][2] > 333.99 && PI.IntervalTable[contii][2] < 334.01) {
				int aaaaaa = 1;
			}
			contii++;

		}
		*/
	}
	//conttotal = conti;
	conttotal = contii;
	//Calc Tcin/Tcout
	for (conti = 0; conti <= (contallhPW - 1); conti++) {
		contc = 1;
		while ((PI.cascade1cPW[contc] - PI.cascade1hPW[conti]) > 0) {
			contc++;
		}
		//if (PI.CPdiffcPW[contc - 1] == 0) {
		//	AreaCosts = AreaCosts;
		//}
		if (PI.CPdiffcPW[contc - 1] == 0) {
			TcLM[conti] = PI.AllTcPW[contc];
		}
		else {
			TcLM[conti] = PI.AllTcPW[contc] + (PI.cascade1hPW[conti] - PI.cascade1cPW[contc]) / PI.CPdiffcPW[contc - 1];
		}
		//if (Particle[0].PS.dTmin > 24.51 && Particle[0].PS.dTmin < 24.7) {
		//	printf("%.12f\t = %.12f\t - %.12f\t * %.12f\t\n", TcLM[conti], PI.AllTcPW[contc] + (PI.cascade1hPW[conti]) / PI.CPdiffcPW[contc - 1], 1 / PI.CPdiffcPW[contc - 1], PI.cascade1cPW[contc]);
		//}

		//if (conti <= (contallhPW - 2)) {
		//	if (PI.cascade1hPW[conti] != PI.cascade1hPW[conti + 1]) {
		PI.IntervalTable[conttotal][0] = PI.cascade1hPW[conti];
		PI.IntervalTable[conttotal][1] = PI.AllThPW[conti];
		PI.IntervalTable[conttotal][2] = TcLM[conti];
		PI.IntervalTable[conttotal][3] = (double)PI.IsHU[conti];
		if (PI.IntervalTable[conttotal][2] > 333.99 && PI.IntervalTable[conttotal][2] < 334.01) {
			int aaaaaa = 1;
		}
		conttotal++;
		//	}
		//}
		/*
		else {
			PI.IntervalTable[conttotal][0] = PI.cascade1hPW[conti];
			PI.IntervalTable[conttotal][1] = PI.AllThPW[conti];
			PI.IntervalTable[conttotal][2] = TcLM[conti];
			PI.IntervalTable[conttotal][3] = (double)PI.IsHU[conti];
			if (PI.IntervalTable[conttotal][2] > 333.99 && PI.IntervalTable[conttotal][2] < 334.01) {
				int aaaaaa = 1;
			}
			conttotal++;

		}
		*/
	}

	//Table and Sort

	for (i = 0; i < conttotal - 1; i++) {
		for (j = 0; j < conttotal - 1; j++) {
			if (PI.IntervalTable[j][0] < PI.IntervalTable[j + 1][0]) {
				TempIntervalT[0] = PI.IntervalTable[j][0]; TempIntervalT[1] = PI.IntervalTable[j][1]; TempIntervalT[2] = PI.IntervalTable[j][2]; TempIntervalT[3] = PI.IntervalTable[j][3]; TempIntervalT[4] = PI.IntervalTable[j][4];
				PI.IntervalTable[j][0] = PI.IntervalTable[j + 1][0]; PI.IntervalTable[j][1] = PI.IntervalTable[j + 1][1]; PI.IntervalTable[j][2] = PI.IntervalTable[j + 1][2]; PI.IntervalTable[j][3] = PI.IntervalTable[j + 1][3]; PI.IntervalTable[j][4] = PI.IntervalTable[j + 1][4];
				PI.IntervalTable[j + 1][0] = TempIntervalT[0]; PI.IntervalTable[j + 1][1] = TempIntervalT[1]; PI.IntervalTable[j + 1][2] = TempIntervalT[2]; PI.IntervalTable[j + 1][3] = TempIntervalT[3]; PI.IntervalTable[j + 1][4] = TempIntervalT[4];
			}
			else if (PI.IntervalTable[j][0] == PI.IntervalTable[j + 1][0] && PI.IntervalTable[j + 1][1] < PI.IntervalTable[j + 1][2]) {
				int aaa = 1;
			}
		}
	}
	ii = 0;
	ii0 = 0;
	iii = 0;
	ii1 = 0;
	jj = 0;
	while (ii < conttotal - 1) {
		if ((PI.IntervalTable[ii][0] - PI.IntervalTable[ii + 1][0]) < 0.00000000001) {//}&& PI.IntervalTable[ii][3] == 0) {
			ii0 = ii;
			ii1 = ii;
			iii = ii;
			while ((PI.IntervalTable[iii][0] - PI.IntervalTable[iii + 1][0]) < 0.00000000001 && iii < conttotal - 1) {// && PI.IntervalTable[iii + 1][4] == 0) {
				ii1++;
				iii++;
			}
			//if (PI.IntervalTable[ii0][3] == 0) {
			for (i = ii0; i <= ii1 - 1; i++) {
				for (j = ii0; j <= ii1 - 1; j++) {
					if (PI.IntervalTable[j][1] < PI.IntervalTable[j + 1][1]) {
						TempIntervalT[1] = PI.IntervalTable[j][1];
						PI.IntervalTable[j][1] = PI.IntervalTable[j + 1][1];
						PI.IntervalTable[j + 1][1] = TempIntervalT[1];
					}
				}
			}
			//}
			//if (PI.IntervalTable[ii1][4] == 0) {
			for (i = ii0; i <= ii1 - 1; i++) {
				for (j = ii0; j <= ii1 - 1; j++) {
					if (PI.IntervalTable[j][2] < PI.IntervalTable[j + 1][2]) {
						TempIntervalT[2] = PI.IntervalTable[j][2];
						PI.IntervalTable[j][2] = PI.IntervalTable[j + 1][2];
						PI.IntervalTable[j + 1][2] = TempIntervalT[2];
					}
				}
			}
			//}
			if (ii0 != ii1) {
				intervalstoreorder[0][jj] = ii0;
				intervalstoreorder[1][jj] = ii1;
				jj++;
			}

			ii = ii1;
			if (ii == ii1) {
				ii++;
			}
		}
		else {
			ii++;
		}
	}

	TotalAreaIntervals = conttotal;

	TotalArea = 0;

	for (cont = 0; cont <= TotalAreaIntervals - 2; cont++) {
		sumqhh = 0;
		sumqhc = 0;
		AreaPen = 0;
		if ((PI.IntervalTable[cont][1] - PI.IntervalTable[cont][2]) < EMAT && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {// - 0.0000000001) {
			AreaPen = 1;
			PI.TPenCosts = PI.TPenCosts + 100 * taCtroc + tbCtroc * abs(PI.IntervalTable[cont][1] - PI.IntervalTable[cont][2]);

		}
		if ((PI.IntervalTable[cont + 1][1] - PI.IntervalTable[cont + 1][2]) < EMAT && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {//} - 0.0000000001) {
			AreaPen = 1;
			PI.TPenCosts = PI.TPenCosts + 100 * taCtroc + tbCtroc * abs(PI.IntervalTable[cont + 1][1] - PI.IntervalTable[cont + 1][2]);
		}
		if (AreaPen == 0) {
			//double sumqhhreport[INTER][7] = {};
			//int contxx = 0;
			if (abs((PI.IntervalTable[cont][1] - PI.IntervalTable[cont][2]) - (PI.IntervalTable[cont + 1][1] - PI.IntervalTable[cont + 1][2])) > 0 && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {
				LMTDk[cont] = ((PI.IntervalTable[cont][1] - PI.IntervalTable[cont][2]) - (PI.IntervalTable[cont + 1][1] - PI.IntervalTable[cont + 1][2])) / log(((PI.IntervalTable[cont][1] - PI.IntervalTable[cont][2]) / (PI.IntervalTable[cont + 1][1] - PI.IntervalTable[cont + 1][2])));
			}
			else {
				LMTDk[cont] = (PI.IntervalTable[cont][1] - PI.IntervalTable[cont][2]);
			}
			for (conthu = 0; conthu <= CS.TotalHU - 1; conthu++) {
				if (((CS.Thuin[conthu] >= PI.IntervalTable[cont][1]) && (CS.Thuout[conthu] <= PI.IntervalTable[cont + 1][1])) && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {
					//PI.PresentHotStreams[cont][0] = 1;
					if ((CS.Thuin[conthu] >= PI.IntervalTable[cont][1]) && (CS.Thuout[conthu] <= PI.IntervalTable[cont + 1][1])) {
						PI.qexch[conth][cont] = CS.CPhu[conthu] * (PI.IntervalTable[cont][1] - PI.IntervalTable[cont + 1][1]);
					}
					else if ((CS.Thuin[conthu] < PI.IntervalTable[cont][1]) && (CS.Thuout[conthu] <= PI.IntervalTable[cont + 1][1])) {
						PI.qexch[conth][cont] = CS.CPhu[conthu] * (CS.Thuin[conthu] - PI.IntervalTable[cont + 1][1]);
					}
					else if ((CS.Thuin[conthu] >= PI.IntervalTable[cont][1]) && (PI.TableThout[conth] > PI.IntervalTable[cont + 1][1])) {
						PI.qexch[conth][cont] = CS.CPhu[conthu] * (PI.IntervalTable[cont][1] - CS.Thuout[conthu]);
					}
					PI.qexch[conth][cont] = CS.CPhu[conthu] * (PI.IntervalTable[cont][1] - PI.IntervalTable[cont + 1][1]);

					sumqhh = sumqhh + PI.qexch[conth][cont] / CS.hhu[conthu];
					/*
					sumqhhreport[contxx][0] = sumqhh;
					sumqhhreport[contxx][1] = PI.TableThin2[conth];
					sumqhhreport[contxx][2] = PI.IntervalTable[cont][1];
					sumqhhreport[contxx][3] = PI.TableThout2[conth];
					sumqhhreport[contxx][4] = PI.IntervalTable[cont + 1][1];
					sumqhhreport[contxx][5] = PI.IntervalTable[cont][0];
					sumqhhreport[contxx][6] = PI.IntervalTable[cont + 1][0];
					contxx++;
					*/
					//PI.qexch[conth][cont] = (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0]);
				}
			}
			//printf("%.6f\t", PI.qexch[conth][cont]);

			for (conth = 0; conth <= nhot - 1; conth++) {
				if (((PI.TableThin2[conth] >= PI.IntervalTable[cont][1]) && (PI.TableThout2[conth] <= PI.IntervalTable[cont + 1][1])) && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {
					//PI.PresentHotStreams[cont][conth + 1] = 1;
					//HEAT EXCHANGED BY THIS STREAM???
					if ((PI.TableThin2[conth] >= PI.IntervalTable[cont][1]) && (PI.TableThout2[conth] <= PI.IntervalTable[cont + 1][1])) {
						PI.qexch[conth][cont] = PI.TableCPh[conth] * (PI.IntervalTable[cont][1] - PI.IntervalTable[cont + 1][1]);
					}
					else if ((PI.TableThin[conth] < PI.IntervalTable[cont][1]) && (PI.TableThout[conth] <= PI.IntervalTable[cont + 1][1])) {
						PI.qexch[conth][cont] = PI.TableCPh[conth] * (PI.TableThin[conth] - PI.IntervalTable[cont + 1][1]);
					}
					else if ((PI.TableThin[conth] >= PI.IntervalTable[cont][1]) && (PI.TableThout[conth] > PI.IntervalTable[cont + 1][1])) {
						PI.qexch[conth][cont] = PI.TableCPh[conth] * (PI.IntervalTable[cont][1] - PI.TableThout[conth]);
					}
					//printf("%.6f\t", PI.qexch[conth][cont]);
					sumqhh = sumqhh + PI.qexch[conth][cont] / PI.Tablehh[conth];
					/*
					sumqhhreport[contxx][0] = sumqhh;
					sumqhhreport[contxx][1] = PI.TableThin2[conth];
					sumqhhreport[contxx][2] = PI.IntervalTable[cont][1];
					sumqhhreport[contxx][3] = PI.TableThout2[conth];
					sumqhhreport[contxx][4] = PI.IntervalTable[cont + 1][1];
					sumqhhreport[contxx][5] = PI.IntervalTable[cont][0];
					sumqhhreport[contxx][6] = PI.IntervalTable[cont + 1][0];
					contxx++;
					*/
				}
				else {
					//PI.PresentHotStreams[cont][conth + 1] = 0;
				}
				//printf("%.6f\t", PI.qexch[conth][cont]);

			}
			//printf("\n\n");
			for (contcu = 0; contcu <= CS.TotalCU - 1; contcu++) {
				if ((CS.Tcuout[contcu] >= PI.IntervalTable[cont][2]) && (CS.Tcuin[contcu] <= PI.IntervalTable[cont + 1][2]) && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {
					//PI.PresentColdStreams[cont][0] = 1;
					if ((CS.Tcuout[contcu] >= PI.IntervalTable[cont][2]) && (CS.Tcuin[contcu] <= PI.IntervalTable[cont + 1][2])) {
						PI.qexcc[contc][cont] = CS.CPcu[contcu] * (PI.IntervalTable[cont][2] - PI.IntervalTable[cont + 1][2]);
					}
					else if ((PI.TableTcout[contc] < PI.IntervalTable[cont][2]) && (PI.TableTcin[contc] <= PI.IntervalTable[cont + 1][2])) {
						PI.qexcc[contc][cont] = CS.CPcu[contcu] * (PI.IntervalTable[cont + 1][2] - CS.Tcuout[contcu]);
					}
					else if ((PI.TableTcout[contc] >= PI.IntervalTable[cont][2]) && (PI.TableTcin[contc] > PI.IntervalTable[cont + 1][2])) {
						PI.qexcc[contc][cont] = CS.CPcu[contcu] * (CS.Tcuin[contcu] - PI.IntervalTable[cont][2]);
					}
					//printf("%.6f\t", PI.qexcc[contc][cont]);
					//PI.qexcc[contc][cont] = CS.CPcu * (PI.IntervalTable[cont][2] - PI.IntervalTable[cont + 1][2]);
					sumqhc = sumqhc + PI.qexcc[contc][cont] / CS.hcu[contcu];
					//if (sumqhc < 0)
					//	AreaCosts = AreaCosts;
				}
				//printf("%.6f\t", PI.qexcc[contc][cont]);
			}
			for (contc = 0; contc <= ncold - 1; contc++) {
				if (((PI.TableTcout2[contc] >= PI.IntervalTable[cont][2]) && (PI.TableTcin2[contc] <= PI.IntervalTable[cont + 1][2])) && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {
					//PI.PresentColdStreams[cont][contc + 1] = 1;
					if ((PI.TableTcout2[contc] >= PI.IntervalTable[cont][2]) && (PI.TableTcin2[contc] <= PI.IntervalTable[cont + 1][2])) {
						PI.qexcc[contc][cont] = PI.TableCPc[contc] * (PI.IntervalTable[cont][2] - PI.IntervalTable[cont + 1][2]);
					}
					else if ((PI.TableTcout[contc] < PI.IntervalTable[cont][2]) && (PI.TableTcin[contc] <= PI.IntervalTable[cont + 1][2])) {
						PI.qexcc[contc][cont] = PI.TableCPc[contc] * (PI.IntervalTable[cont + 1][2] - PI.TableTcout[contc]);
					}
					else if ((PI.TableTcout[contc] >= PI.IntervalTable[cont][2]) && (PI.TableTcin[contc] > PI.IntervalTable[cont + 1][2])) {
						PI.qexcc[contc][cont] = PI.TableCPc[contc] * (PI.IntervalTable[cont][2] - PI.TableTcin[contc]);
					}

					//printf("%.6f\t", PI.qexcc[contc][cont]);
					//if (PI.qexcc[contc][cont] < 0)
						//AreaCosts = AreaCosts;
					sumqhc = sumqhc + PI.qexcc[contc][cont] / PI.Tablehc[contc];
					//if (sumqhc < 0)
						//AreaCosts = AreaCosts;
				}
				else {
					//PI.PresentColdStreams[cont][contc + 1] = 0;
				}
				//printf("%.6f\t", PI.qexcc[contc][cont]);

			}
			//printf("\n");

			//if (AreaPen == 0) {
			//	PI.Areak[cont] = 10000000;
			//	TotalArea = TotalArea + PI.Areak[cont];
			//}
			//else {
			//if (cont == 26) {
				//AreaCosts = AreaCosts;
			//}

			PI.Areak[cont] = (1 / LMTDk[cont]) * (sumqhh + sumqhc);
			if (isnan(PI.Areak[cont]) == 1) {
				PI.Areak[cont] = 99999999999999;
				PI.TPenCosts = PI.TPenCosts + 10000000;
			}
			if (AreaPen > 0) {
				PI.Areak[cont] = 99999999999999;
				PI.TPenCosts = PI.TPenCosts + 10000000;
			}
			//if (PI.Areak[cont] > 0 && PI.IntervalTable[cont][0] < 0.001) {
			//	AreaCosts = AreaCosts;
			//}
			if (PI.Areak[cont] < 0)
				PI.Areak[cont] = 0;
			TotalArea = TotalArea + PI.Areak[cont];

		}
		if (cont == 507) {
			TotalArea = TotalArea;
		}

	}


	/*
	//if (printactivated == 1) {
	for (cont = 0; cont <= TotalAreaIntervals - 2; cont++) {

		for (conth = 0; conth <= nhot - 1; conth++) {
			if (((PI.TableThin2[conth] >= PI.IntervalTable[cont][1]) && (PI.TableThout2[conth] <= PI.IntervalTable[cont + 1][1]))) {// && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0)) {
				zh[conth][cont] = 1;
			}
		}
		for (conthu = 0; conthu <= CS.TotalHU - 1; conthu++) {
			if (((CS.Thuin[conthu] >= PI.IntervalTable[cont][1]) && (CS.Thuout[conthu] <= PI.IntervalTable[cont + 1][1]))) {// && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0)) {
				zh[nhot - 1 + conthu][cont] = 1;
			}
		}
		for (contcu = 0; contcu <= CS.TotalCU - 1; contcu++) {
			if ((CS.Tcuout[contcu] >= PI.IntervalTable[cont][2]) && (CS.Tcuin[contcu] <= PI.IntervalTable[cont + 1][2])) {// && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0)) {
				zc[contcu][cont] = 1;
			}
		}
		for (contc = (0 + CS.TotalCU - 1); contc <= (CS.TotalCU - 1 + ncold - 1); contc++) {
			if (((PI.TableTcout2[contc] >= PI.IntervalTable[cont][2]) && (PI.TableTcin2[contc] <= PI.IntervalTable[cont + 1][2]))) {// && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0)) {
				zc[contc][cont] = 1;
			}
			//Fh[conth][contc][cont] = PI.TableCPc[contc] / (sumcpc[cont]);
			//Fc[conth][contc][cont] = PI.TableCPh[conth] / (sumcph[cont]);
			//Q[conth][contc][cont] = PI.TableCPh[conth] / (sumcph[cont]);
		}

	}
	for (cont = 0; cont <= TotalAreaIntervals - 2; cont++) {
		for (conth = 0; conth <= nhot; conth++) {
			for (contc = 0; contc <= ncold; contc++) {
				if (zh[conth][cont] == 1 && zc[contc][cont] == 1) {
					z0[conth][contc][cont] = 1;
				}
			}
		}
	}
	kfinal = 0;
	kexists = 0;
	for (cont = 0; cont <= TotalAreaIntervals - 2; cont++) {
		kexists = 0;
		for (conth = 0; conth <= nhot; conth++) {
			for (contc = 0; contc <= ncold; contc++) {
				if (zh[conth][cont] == 1 && zc[contc][cont] == 1) {
					kexists = 1;
				}
			}
		}
		if (kexists == 1) {
			for (conth = 0; conth <= nhot; conth++) {
				for (contc = 0; contc <= ncold; contc++) {
					if (zh[conth][cont] == 1 && zc[contc][cont] == 1) {
						z[conth][contc][kfinal] = 1;
					}
				}
			}
			kfinal++;
		}
	}
	//}
	*/

	int noofHU = 0;
	int noofCU = 0;

	for (conthu = 0; conthu <= CS.TotalHUTemp - 1; conthu++) { // totalhu dever� ser o numero original
		PI.HU[conthu] = PS.HUFrac[conthu] * HUtotal;
		if (PI.HU[conthu] > 0.0001) {
			noofHU++;
		}
	}
	for (contcu = 0; contcu <= CS.TotalCUTemp - 1; contcu++) {
		PI.CU[contcu] = PS.CUFrac[contcu] * CUtotal;
		if (PI.CU[contcu] > 0.0001) {
			noofCU++;
		}
	}

	if (isnan(TotalArea) == 1) {
		PI.TPenCosts = PI.TPenCosts + 100 * taCtroc;
		HECC = PI.TPenCosts;
	}
	else {
		//PI.Nun = (numstreams - 1.0);
		PI.Nun = (CS.noofstreams - 1.0 + noofHU + noofCU);
	}

	vector<double> phi; phi.resize(CS.TotalStreams);
	vector<double> phiHU; phiHU.resize(CS.TotalHU);
	vector<double> phiCU; phiCU.resize(CS.TotalCU);

	for (int i = 0; i <= CS.TotalStreams - 1; i++) {
		if (CS.C[i] > 0.001) {
			phi[i] = (pow((CS.C0 / CS.C[i]), 1 / CS.beta0) / pow(PI.Nun, (1 - CS.beta[i] / CS.beta0))) * pow(TotalArea, (1 - CS.beta[i] / CS.beta0));
		}
		else {
			phi[i] = 1;
		}
	}
	for (int i = 0; i <= CS.TotalHU - 1; i++) { // totalhu dever� ser o numero original	
		if (CS.CHU[i] > 0.001) {
			phiHU[i] = (pow((CS.C0 / CS.CHU[i]), 1 / CS.beta0) / pow(PI.Nun, (1 - CS.betaHU[i] / CS.beta0))) * pow(TotalArea, (1 - CS.betaHU[i] / CS.beta0));
		}
		else {
			phiHU[i] = 1;
		}
	}
	for (int i = 0; i <= CS.TotalCU - 1; i++) {
		if (CS.CCU[i] > 0.001) {
			phiCU[i] = (pow((CS.C0 / CS.CCU[i]), 1 / CS.beta0) / pow(PI.Nun, (1 - CS.betaCU[i] / CS.beta0))) * pow(TotalArea, (1 - CS.betaCU[i] / CS.beta0));
		}
		else {
			phiCU[i] = 1;
		}
	}
	int ss = 0;
	int conthh = 0;
	int contcc = 0;

	for (ss = 0; ss <= CS.TotalStreams - 1; ss++) {
		if (CS.Tin[ss] > CS.Tout[ss] && CS.CP[ss] > 0.0) {
			PI.Tablephih[conthh] = phi[ss];
			conthh++;
		}
		else if (CS.Tin[ss] < CS.Tout[ss] && CS.CP[ss] > 0.0) {
			PI.Tablephic[contcc] = phi[ss];
			contcc++;
		}
	}

	//AREA CORRIGIDA
	TotalArea = 0;

	for (cont = 0; cont <= TotalAreaIntervals - 2; cont++) {
		sumqhh = 0;
		sumqhc = 0;
		AreaPen = 0;
		if (AreaPen == 0) {
			for (conthu = 0; conthu <= CS.TotalHU - 1; conthu++) {
				if (((CS.Thuin[conthu] >= PI.IntervalTable[cont][1]) && (CS.Thuout[conthu] <= PI.IntervalTable[cont + 1][1])) && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {

					sumqhh = sumqhh + PI.qexch[conth][cont] / (phiHU[conthu] * CS.hhu[conthu]);

				}
			}

			for (conth = 0; conth <= nhot - 1; conth++) {
				if (((PI.TableThin2[conth] >= PI.IntervalTable[cont][1]) && (PI.TableThout2[conth] <= PI.IntervalTable[cont + 1][1])) && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {

					sumqhh = sumqhh + PI.qexch[conth][cont] / (PI.Tablephih[conth] * PI.Tablehh[conth]);

				}


			}
			//printf("\n\n");
			for (contcu = 0; contcu <= CS.TotalCU - 1; contcu++) {
				if ((CS.Tcuout[contcu] >= PI.IntervalTable[cont][2]) && (CS.Tcuin[contcu] <= PI.IntervalTable[cont + 1][2]) && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {
					sumqhc = sumqhc + PI.qexcc[contc][cont] / (phiCU[contcu] * CS.hcu[contcu]);
				}
			}
			for (contc = 0; contc <= ncold - 1; contc++) {
				if (((PI.TableTcout2[contc] >= PI.IntervalTable[cont][2]) && (PI.TableTcin2[contc] <= PI.IntervalTable[cont + 1][2])) && (PI.IntervalTable[cont][0] - PI.IntervalTable[cont + 1][0] > 0.0001)) {
					sumqhc = sumqhc + PI.qexcc[contc][cont] / (PI.Tablephic[contc] * PI.Tablehc[contc]);
				}
			}

			PI.Areak[cont] = (1 / LMTDk[cont]) * (sumqhh + sumqhc);
			if (isnan(PI.Areak[cont]) == 1) {
				PI.Areak[cont] = 99999999999999;
			}
			if (AreaPen > 0) {
				PI.Areak[cont] = 99999999999999;
			}
			if (PI.Areak[cont] < 0)
				PI.Areak[cont] = 0;
			TotalArea = TotalArea + PI.Areak[cont];

		}
	}


	PI.intervals = TotalAreaIntervals - 2;

	double end = clock();
	double TotalTime = (double)(end - start) / CLOCKS_PER_SEC;

	if (print == 1) {







		//double PS.TotalOC = 0;
		conthu = 0;
		contcu = 0;
		PS.TotalOC = 0;

		for (conthu = 0; conthu <= CS.TotalHUTemp - 1; conthu++) {
			PS.TotalOC = PS.TotalOC + PI.HU[conthu] * CS.HUcosts[conthu];
		}
		for (contcu = 0; contcu <= CS.TotalCUTemp - 1; contcu++) {
			PS.TotalOC = PS.TotalOC + PI.CU[contcu] * CS.CUcosts[contcu];
		}

		PS.TotalCC = 0;

		PS.TotalCC = PI.Nun * (CS.B0 + CS.C0 * pow((TotalArea / PI.Nun), CS.beta0));


		char buffer1[80]{};
		char buffer2[200]{};

		sprintf_s(buffer1, "%i-%i-%i Pinch %ih%im%is %i.txt", today[0], today[1], today[2], today[3], today[4], today[5], (int)(PS.TotalOC + PS.TotalCC + PI.TPenCosts));

		ofstream myfile(buffer1);

		sprintf_s(buffer2, "PS.dTmin = %.6f\n\n", PS.dTmin);
		myfile << buffer2;
		sprintf_s(buffer2, "HUFRAC\tFLOWRATE\tTotalEnthalpy\tTout\n");
		myfile << buffer2;
		for (int i = 0; i <= CS.TotalHUTemp - 1; i++) {
			sprintf_s(buffer2, "%.6f\t%.6f\t%.6f\t%.6f\n", PS.HUFrac[i], PI.flowrateHU[i], PI.HU[i], PS.VarHUTempOut[i]);
			myfile << buffer2;
		}
		sprintf_s(buffer2, "CUFRAC\tFLOWRATE\tTotalEnthalpy\tTout\n");
		myfile << buffer2;
		for (int i = 0; i <= CS.TotalCUTemp - 1; i++) {
			sprintf_s(buffer2, "%.6f\t%.6f\t%.6f\t%.6f\n", PS.CUFrac[i], PI.flowrateCU[i], PI.CU[i], PS.VarCUTempOut[i]);
			myfile << buffer2;
		}

		sprintf_s(buffer2, "\nGCC cascade\nH\tT\n");
		myfile << buffer2;
		int AA = PI.cascade2.size();
		for (int i = 0; i <= AA - 1; i++) {
			sprintf_s(buffer2, "%.6f\t%.6f\n", PI.cascade2[i], PI.AllTempFinal[i]);
			myfile << buffer2;
			if (PI.cascade2[i] == 0) {
				break;
			}
		}

		sprintf_s(buffer2, "\n\nHot cascade\n");
		myfile << buffer2;
		AA = PI.cascade1h.size();
		for (int i = 0; i <= AA - 1; i++) {
			sprintf_s(buffer2, "%.6f\t%.6f\n", PI.cascade1h[i], PI.AllThFinal[i]);
			myfile << buffer2;
			if (PI.cascade1h[i] == 0) {
				break;
			}
		}
		sprintf_s(buffer2, "\n\nCold cascade\n");
		myfile << buffer2;
		AA = PI.cascade1c.size();
		for (int i = 0; i <= AA - 1; i++) {
			sprintf_s(buffer2, "%.6f\t%.6f\n", PI.cascade1c[i], PI.AllTcFinal[i]);
			myfile << buffer2;
			if (PI.cascade1c[i] == 0) {
				break;
			}
		}

		int s2 = 0;

		int conth2 = 0;
		int contallh2 = 0;
		int contc2 = 0;
		int contallc2 = 0;
		int contall2 = contall;
		int numstreams2 = 0;
		//double start = clock();
		for (conthu = 0; conthu <= CS.TotalHU - 1; conthu++) {
			PI.AllTemp[contall + 2 * conthu] = CS.Thuin[conthu] - PS.dTmin / 2;
			PI.AllTemp[contall + 2 * conthu + 1] = CS.Thuout[conthu] - PS.dTmin / 2;
			contall2 = contall2 + 2;
		}
		for (contcu = 0; contcu <= CS.TotalCU - 1; contcu++) {
			PI.AllTemp[(2 * CS.TotalHU + contall + 2 * contcu)] = CS.Tcuin[contcu] + PS.dTmin / 2;
			PI.AllTemp[(2 * CS.TotalHU + contall + 2 * contcu + 1)] = CS.Tcuout[contcu] + PS.dTmin / 2;
			contall2 = contall2 + 2;
		}

		//BALANCED grand composite

		sort(PI.AllTemp.begin(), PI.AllTemp.begin() + contall2, greater<double>());

		cont2 = 0;
		for (cont = 0; cont <= contall2 - 1; cont++) {
			if (abs(PI.AllTemp[cont] - PI.AllTemp[cont + 1]) >= 0) {
				PI.AllTempFinal[cont2] = PI.AllTemp[cont];
				cont2++;
			}
		}

		numinter = cont2;

		PI.dT[0] = 0;
		PI.CPdiff[0] = 0;
		PI.dH[0] = 0;
		PI.cascade1[0] = 0;

		sumcph = 0;
		sumcpc = 0;

		for (conti = 1; conti <= numinter - 1; conti++) {
			PI.dT[conti] = PI.AllTempFinal[conti - 1] - PI.AllTempFinal[conti];
			sumcph = 0;
			sumcpc = 0;
			for (conth = 0; conth <= nhot - 1; conth++) {
				if ((PI.TableThin[conth] - PI.AllTempFinal[conti] > 0) && (PI.TableThout[conth] - PI.AllTempFinal[conti] <= -0)) {
					sumcph = sumcph + PI.TableCPh[conth];
				}
			}
			for (conthu = 0; conthu <= CS.TotalHU - 1; conthu++) {
				if (((CS.Thuin[conthu] - PS.dTmin / 2) - PI.AllTempFinal[conti] > 0) && ((CS.Thuout[conthu] - PS.dTmin / 2) - PI.AllTempFinal[conti] <= -0)) {
					sumcph = sumcph + CS.CPhu[conthu]; // PS.HUFrac[conthu] * HUtotal / (CS.Thuin[conthu] - CS.Thuout[conthu]);//WRONG
				}
			}
			for (contc = 0; contc <= ncold - 1; contc++) {
				if ((PI.TableTcout[contc] - PI.AllTempFinal[conti] > 0) && (PI.TableTcin[contc] - PI.AllTempFinal[conti] <= -0)) {
					sumcpc = sumcpc + PI.TableCPc[contc];
				}
			}
			for (contcu = 0; contcu <= CS.TotalCU - 1; contcu++) {
				if (((CS.Tcuout[contcu] + PS.dTmin / 2) - PI.AllTempFinal[conti] > 0) && ((CS.Tcuin[contcu] + PS.dTmin / 2) - PI.AllTempFinal[conti] <= -0)) {
					sumcpc = sumcpc + CS.CPcu[contcu];//PS.CUFrac[contcu] * CUtotal / (CS.Tcuout[contcu] - CS.Tcuin[contcu]);
				}
			}
			PI.CPdiff[conti] = sumcpc - sumcph;
			PI.dH[conti] = PI.CPdiff[conti] * PI.dT[conti];
			PI.cascade1[conti] = PI.cascade1[conti - 1] - PI.dH[conti];
		}

		//maxcascade = abs(*max_element(PI.cascade1, PI.cascade1 + numinter));
		mincascade = abs(*min_element(PI.cascade1.begin(), PI.cascade1.begin() + numinter));

		//if (maxcascade < mincascade) {
		//	maxcascade = mincascade;
		//}
		PI.cascade2[0] = mincascade;
		for (conti = 1; conti <= numinter - 1; conti++) {
			PI.cascade2[conti] = PI.cascade2[conti - 1] - PI.dH[conti];
		}

		sprintf_s(buffer2, "\nBGCC cascade\nH\tT\n");
		myfile << buffer2;
		AA = PI.cascade2.size();
		for (int i = 0; i <= AA - 1; i++) {
			sprintf_s(buffer2, "%.6f\t%.6f\n", PI.cascade1[i], PI.AllTempFinal[i]);
			myfile << buffer2;
			if (PI.cascade2[i] == 0) {
				break;
			}
		}


		sprintf_s(buffer2, "\n\nHot balanced cascade\n");
		myfile << buffer2;
		AA = PI.cascade1hPW.size();
		for (int i = 0; i <= AA - 1; i++) {
			sprintf_s(buffer2, "%.6f\t%.6f\n", PI.cascade1hPW[i], PI.AllThPW[i]);
			myfile << buffer2;
			if (PI.cascade1hPW[i] == 0) {
				break;
			}
		}



		sprintf_s(buffer2, "\n\nCold balanced cascade\n");
		myfile << buffer2;
		AA = PI.cascade1cPW.size();
		for (int i = 0; i <= AA - 1; i++) {
			sprintf_s(buffer2, "%.6f\t%.6f\n", PI.cascade1cPW[i], PI.AllTcPW[i]);
			myfile << buffer2;
			if (PI.cascade1cPW[i] == 0) {
				break;
			}
		}

		sprintf_s(buffer2, "\n\nPresent Hot Util\n");
		myfile << buffer2;
		AA = HUPresent.size();
		for (int ii = 0; ii <= HUPresent[i].size() - 1; ii++) {
			sprintf_s(buffer2, "%.6f\t", CS.Thuin[ii]);
			myfile << buffer2;
		}
		sprintf_s(buffer2, "\n");
		myfile << buffer2;
		for (int ii = 0; ii <= HUPresent[i].size() - 1; ii++) {
			sprintf_s(buffer2, "%.6f\t", CS.Thuout[ii]);
			myfile << buffer2;
		}
		sprintf_s(buffer2, "\n");
		myfile << buffer2;
		for (int i = 0; i <= AA - 1; i++) {
			for (int ii = 0; ii <= HUPresent[i].size() - 1; ii++) {
				sprintf_s(buffer2, "%i\t", HUPresent[i][ii]);
				myfile << buffer2;
			}
			sprintf_s(buffer2, "\n");
			myfile << buffer2;
			if (PI.cascade1hPW[i] == 0) {
				break;
			}
		}

		sprintf_s(buffer2, "\n\nPresent Cold Util\n");
		myfile << buffer2;
		AA = CUPresent.size();
		for (int ii = 0; ii <= CUPresent[i].size() - 1; ii++) {
			sprintf_s(buffer2, "%.6f\t", CS.Tcuin[ii]);
			myfile << buffer2;
		}
		sprintf_s(buffer2, "\n");
		myfile << buffer2;
		for (int ii = 0; ii <= CUPresent[i].size() - 1; ii++) {
			sprintf_s(buffer2, "%.6f\t", CS.Tcuout[ii]);
			myfile << buffer2;
		}
		sprintf_s(buffer2, "\n");
		myfile << buffer2;
		for (int i = 0; i <= AA - 1; i++) {
			for (int ii = 0; ii <= CUPresent[i].size() - 1; ii++) {
				sprintf_s(buffer2, "%i\t", CUPresent[i][ii]);
				myfile << buffer2;
			}
			sprintf_s(buffer2, "\n");
			myfile << buffer2;
			if (PI.cascade1cPW[i] == 0) {
				break;
			}
		}

		sprintf_s(buffer2, "\n\nIntervalTable\n");
		myfile << buffer2;
		AA = PI.IntervalTable.size();
		for (int i = 0; i <= AA - 1; i++) {
			sprintf_s(buffer2, "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", PI.IntervalTable[i][0], PI.IntervalTable[i][1], PI.IntervalTable[i][2], PI.IntervalTable[i][3], PI.IntervalTable[i][4]);
			myfile << buffer2;
		}

		sprintf_s(buffer2, "\nTotalCost = %.2f \nCC = %.2f\nOC = %.2f\nTotalTime = %.2f", PS.TotalOC + PS.TotalCC + PI.TPenCosts, PS.TotalCC, PS.TotalOC, TotalTime);
		myfile << buffer2;

		myfile.close();
		AA = 0;
	}


}

void BuildPinchCS(SolutionStruct Solution, CaseStudyHENStruct CSHEN, CaseStudyPinch& CSPinch, CaseStudyPinch CSzero) {
	CSPinch = CSzero;

	for (int i = 0; i < CSHEN.Streams.size(); i++) {
		CSPinch.Tin.push_back(CSHEN.Streams[i][1]);
		CSPinch.Tout.push_back(CSHEN.Streams[i][2]);
		CSPinch.CP.push_back(CSHEN.Streams[i][3]);
		CSPinch.h.push_back(CSHEN.Streams[i][4]);
	}
	CSPinch.B = CSHEN.B;
	CSPinch.C = CSHEN.C;
	CSPinch.beta = CSHEN.beta;

	int hotstreamind = 0;
	int coldstreamind = 0;

	CSPinch.noofstreams = CSPinch.Tin.size();// hotstreamind + coldstreamind;
	CSPinch.TotalStreams = CSPinch.noofstreams;

	CSPinch.TotalHU = CSHEN.Thuin.size();
	CSPinch.TotalCU = CSHEN.Tcuin.size();

	CSPinch.TotalHUTemp = CSPinch.TotalHU;
	CSPinch.TotalCUTemp = CSPinch.TotalCU;

	CSPinch.Thuin.resize(CSPinch.TotalHU); CSPinch.Thuout.resize(CSPinch.TotalHU); CSPinch.CPhu.resize(CSPinch.TotalHU); CSPinch.hhu.resize(CSPinch.TotalHU); CSPinch.CorrespNumHU.resize(CSPinch.TotalHU);
	CSPinch.Tcuin.resize(CSPinch.TotalCU); CSPinch.Tcuout.resize(CSPinch.TotalCU); CSPinch.CPcu.resize(CSPinch.TotalCU); CSPinch.hcu.resize(CSPinch.TotalCU); CSPinch.CorrespNumCU.resize(CSPinch.TotalCU);

	for (int i = 0; i <= CSPinch.TotalHU - 1; i++) {
		CSPinch.Thuin[i] = CSHEN.HUStreams[i][1];// Thuin[i];
		CSPinch.Thuout[i] = CSHEN.HUStreams[i][2]; //CSHEN.Thuout[i];
		CSPinch.CPhu[i] = 1.0;
		CSPinch.hhu[i] = CSHEN.HUStreams[i][4]; //CSHEN.hhu[i];
		CSPinch.HUcosts.push_back(CSHEN.HUCosts[i]);
		CSPinch.CorrespNumHU[i] = i + 1;
		CSPinch.BHU.push_back(CSHEN.Bh[i]);
		CSPinch.CHU.push_back(CSHEN.Ch[i]);
		CSPinch.betaHU.push_back(CSHEN.betah[i]);
	}
	for (int i = 0; i <= CSPinch.TotalCU - 1; i++) {
		CSPinch.Tcuin[i] = CSHEN.CUStreams[i][1]; //CSHEN.Tcuin[i];
		CSPinch.Tcuout[i] = CSHEN.CUStreams[i][2]; //CSHEN.Tcuout[i];
		CSPinch.CPcu[i] = 1.0;
		CSPinch.hcu[i] = CSHEN.CUStreams[i][4]; //CSHEN.hcu[i];
		CSPinch.CUcosts.push_back(CSHEN.CUCosts[i]);
		CSPinch.CorrespNumCU[i] = i + 1;
		CSPinch.BCU.push_back(CSHEN.Bc[i]);
		CSPinch.CCU.push_back(CSHEN.Cc[i]);
		CSPinch.betaCU.push_back(CSHEN.betac[i]);
	}
	CSPinch.B0 = CSHEN.B0;
	CSPinch.C0 = CSHEN.C0;
	CSPinch.beta0 = CSHEN.beta0;


}

void ResizePinch(CaseStudyPinch CS, PinchSolution& PS, PinchInterm& PI) {

	//Declara��o de vetores referentes a correntes
	int streampieces = CS.Tout.size();
	int maxsizeHU = CS.Thuout.size();
	int maxsizeCU = CS.Tcuout.size();

	//PS.Areak.resize((2 * (streampieces + maxsizeHU + maxsizeCU)));

	int INTER = (2 * (streampieces + maxsizeHU + maxsizeCU)) + 10;

	PI.Areak.resize(INTER);

	PI.cascade1.resize(INTER);
	PI.cascade2.resize(INTER);

	//PS.cascade2.resize(INTER);

	PI.TableThin2.resize(INTER);
	PI.TableThout2.resize(INTER);
	PI.TableTcin2.resize(INTER);
	PI.TableTcout2.resize(INTER);

	PI.AllTemp.resize(INTER);
	PI.AllTempFinal.resize(INTER);

	//PS.AllTempFinal.resize(INTER);

	PI.IntervalTable.resize(INTER);
	for (int i = 0; i <= INTER - 1; i++) {
		PI.IntervalTable[i].resize(5);
	}

	PI.cascade1h.resize(INTER);
	//PS.cascade1h.resize(INTER);
	//PI.cascade2h.resize(INTER);
	PI.cascade1c.resize(INTER);
	//PI.cascade2c.resize(INTER);
	//PS.cascade1c.resize(INTER);

	PI.TableThin.resize(INTER);
	PI.TableThout.resize(INTER);
	PI.TableTcin.resize(INTER);
	PI.TableTcout.resize(INTER);

	PI.AllTh.resize(INTER);
	PI.AllTc.resize(INTER);
	PI.AllThFinal.resize(INTER);
	PI.AllTcFinal.resize(INTER);

	//PS.AllThFinal.resize(INTER);
	//PS.AllTcFinal.resize(INTER);

	PI.AllThPW.resize(INTER);
	PI.AllTcPW.resize(INTER);

	//PS.AllThPW.resize(INTER);
	//PS.AllTcPW.resize(INTER);

	PI.TableCPh.resize(INTER);
	PI.TableCPc.resize(INTER);
	PI.Tablehh.resize(INTER);
	PI.Tablehc.resize(INTER);
	PI.Tablephih.resize(INTER);
	PI.Tablephic.resize(INTER);

	PI.sumcpdtHU.resize(CS.Thuout.size());
	PI.sumcpdtCU.resize(CS.Tcuout.size());

	PS.VarHUTempOut.resize(CS.Thuout.size());
	PS.VarCUTempOut.resize(CS.Tcuout.size());

	int TotalHUTemp = CS.Thuout.size();
	int TotalCUTemp = CS.Tcuout.size();

	PI.HU.resize(TotalHUTemp);
	PI.CU.resize(TotalCUTemp);

	//PS.HU.resize(TotalHUTemp);
	//PS.CU.resize(TotalCUTemp);

	int intervals = 0;
	double TotalCost = 0;

	PS.HUFrac.resize(TotalHUTemp);
	PS.CUFrac.resize(TotalCUTemp);
	PS.VelHUFrac.resize(TotalHUTemp);
	PS.VelCUFrac.resize(TotalCUTemp);

	PS.VarHUTempOut.resize(TotalHUTemp);
	PS.VarCUTempOut.resize(TotalCUTemp);
	PS.VelVarHUTempOut.resize(TotalHUTemp);
	PS.VelVarCUTempOut.resize(TotalCUTemp);


	PI.flowrateHU.resize(TotalHUTemp);
	PI.flowrateCU.resize(TotalCUTemp);

	//novos
	PI.dT.resize(INTER);
	PI.CPdiff.resize(INTER);
	PI.dH.resize(INTER);

	PI.IsCU.resize(INTER);
	PI.IsHU.resize(INTER);

	PI.dTh.resize(INTER);
	PI.dHh.resize(INTER);
	PI.CPdiffh.resize(INTER);

	PI.dTc.resize(INTER);
	PI.dHc.resize(INTER);
	PI.CPdiffc.resize(INTER);

	PI.dTcPW.resize(INTER);
	PI.dThPW.resize(INTER);
	PI.CPdiffcPW.resize(INTER);
	PI.CPdiffhPW.resize(INTER);
	PI.dHcPW.resize(INTER);
	PI.dHhPW.resize(INTER);
	PI.cascade1cPW.resize(INTER);
	PI.cascade1hPW.resize(INTER);

	//PS.cascade1cPW.resize(INTER);
	//PS.cascade1hPW.resize(INTER);

	PI.qexch.resize(INTER);
	PI.qexcc.resize(INTER);
	for (int i = 0; i <= INTER - 1; i++) {
		PI.qexch[i].resize(INTER);
		PI.qexcc[i].resize(INTER);
	}

}

void BuildTrivialPS(CaseStudyHENStruct CaseStudyHEN, SolutionStruct Solution, PinchSolution& TrivialPS) {

	TrivialPS.dTmin = 10;

	double TotalQHU = 0;
	double TotalQCU = 0;

	double Tmin = +INFINITY;
	int Tcuind = 0;
	for (int u = 0; u <= CaseStudyHEN.Tcuin.size() - 1; u++) {
		if (CaseStudyHEN.Tcuin[u] < Tmin) {
			Tcuind = u;
		}
	}
	for (int u = 0; u <= CaseStudyHEN.Tcuin.size() - 1; u++) {
		if (u == Tcuind) {
			TrivialPS.CUFrac[u] = 1.0;
		}
		else {
			TrivialPS.CUFrac[u] = 0.0;
		}
	}


	Tmin = -INFINITY;
	int Thuind = 0;
	for (int u = 0; u <= CaseStudyHEN.Thuin.size() - 1; u++) {
		if (CaseStudyHEN.Thuin[u] > Tmin) {
			Thuind = u;
		}
	}
	for (int u = 0; u <= CaseStudyHEN.Thuin.size() - 1; u++) {
		if (u == Thuind) {
			TrivialPS.HUFrac[u] = 1.0;
		}
		else {
			TrivialPS.HUFrac[u] = 0.0;
		}
	}




}

double OCHI(vector<double> HU, vector<double> CU, vector<double> HUcosts, vector<double> CUcosts) {

	double Total = 0;
	int conthu = 0;
	int contcu = 0;
	Total = 0;

	for (conthu = 0; conthu <= HU.size() - 1; conthu++) {
		Total = Total + HU[conthu] * HUcosts[conthu];
	}
	for (contcu = 0; contcu <= CU.size() - 1; contcu++) {
		Total = Total + CU[contcu] * CUcosts[contcu];
	}

	return Total;
}

double CCHI(int Nun, int intervals, vector<double> Areak, double B, double C, double beta) {
	double Total;
	double TotalArea = 0;
	int cont;
	double AF = 1.0;

	for (cont = 0; cont <= intervals; cont++) {
		TotalArea = TotalArea + Areak[cont];
	}

	Total = Nun * AF * (B + C * pow((TotalArea / Nun), beta));

	return Total;
}


void PSO(int nparticles, double VelFactor, double wmin, double wmax, double c1, double c2, int PSOMaxIter, int Input, SolutionStruct InputSolution, SolutionStruct StructureSolution, PinchInterm PI, CaseStudyStruct CaseStudy, CaseStudyPinch CSPinch, CaseStudyPinch CSPinchzero, SolutionStruct& BestSol, double dTminub, double dTminlb, double fraclb, double fracub, double start, vector<int> today) {

	vector<SolutionStruct> Particle; Particle.resize(nparticles);
	vector<SolutionStruct> ParticleBest; ParticleBest.resize(nparticles);
	SolutionStruct GlobalBest;

	PinchInterm PIzero = PI;

	double BestCostTemp = +INFINITY;
	int improved = 0;
	int BestCostInd = 0;



	for (int p = 0; p <= nparticles - 1; p++) {

		Particle[p] = StructureSolution;

		if (Input == 1 && p == 0) {
			Particle[p] = InputSolution;
		}
		else {
			Particle[p].PS.dTmin = dTminlb + (dTminub - dTminlb) * (double)rand() / (double)RAND_MAX;
			Particle[p].PS.VeldTmin = VelFactor * (dTminub - dTminlb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);

			double sumaux = 0;
			for (int u = 0; u <= CSPinch.Thuin.size() - 1; u++) {
				Particle[p].PS.HUFrac[u] = fraclb + (fracub - fraclb) * (double)rand() / (double)RAND_MAX;
				sumaux = sumaux + Particle[p].PS.HUFrac[u];
			}
			for (int u = 0; u <= CSPinch.Thuin.size() - 1; u++) {
				Particle[p].PS.HUFrac[u] = Particle[p].PS.HUFrac[u] / sumaux;
				Particle[p].PS.VelHUFrac[u] = VelFactor * (fracub - fraclb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
				Particle[p].PS.VarHUTempOut[u] = CSPinch.Thuout[u];
				Particle[p].PS.VelVarHUTempOut[u] = 0;
			}
			sumaux = 0;
			for (int u = 0; u <= CSPinch.Tcuin.size() - 1; u++) {
				Particle[p].PS.CUFrac[u] = fraclb + (fracub - fraclb) * (double)rand() / (double)RAND_MAX;
				sumaux = sumaux + Particle[p].PS.CUFrac[u];
			}
			for (int u = 0; u <= CSPinch.Tcuin.size() - 1; u++) {
				Particle[p].PS.CUFrac[u] = Particle[p].PS.CUFrac[u] / sumaux;
				Particle[p].PS.VelCUFrac[u] = VelFactor * (fracub - fraclb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
				Particle[p].PS.VarCUTempOut[u] = CSPinch.Tcuout[u];
				Particle[p].PS.VelVarCUTempOut[u] = 0;
			}


			//Sys(CaseStudy, Particle[p], 0);
			//BuildHICS(Particle[p], CaseStudy, CS, CSzero);
			PI = PIzero;
			Pinch(CSPinch, Particle[p].PS, PI, 0, start, today);
			Particle[p].TotalCosts = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts) + CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
			Particle[p].TPenCosts = Particle[p].TPenCosts + PI.TPenCosts;
		}
		//cout << Particle[p].SysTACHI; cout << "\n";

		ParticleBest[p] = Particle[p];
		if (Particle[p].TotalCosts < BestCostTemp) {
			BestCostInd = p;
			BestCostTemp = Particle[p].TotalCosts;
			improved = 1;
		}
	}
	if (improved == 1) {
		improved = 0;
		GlobalBest = Particle[BestCostInd];
	}

	improved = 0;
	BestCostInd = 0;

	double r1 = 0;
	double r2 = 0;


	for (int PSOIter = 0; PSOIter <= PSOMaxIter; PSOIter++) {
		improved = 0;
		for (int p = 0; p <= nparticles - 1; p++) {
			r1 = (double)rand() / (double)RAND_MAX;
			r2 = (double)rand() / (double)RAND_MAX;
			Particle[p].PS.VeldTmin = ((wmax - wmin) * exp(-PSOIter / PSOMaxIter) + wmin) * Particle[p].PS.VeldTmin + c1 * r1 * (ParticleBest[p].PS.dTmin - Particle[p].PS.dTmin) + c2 * r2 * (GlobalBest.PS.dTmin - Particle[p].PS.dTmin);
			Particle[p].PS.dTmin = Particle[p].PS.dTmin + Particle[p].PS.VeldTmin;
			if (Particle[p].PS.dTmin > dTminub) {
				Particle[p].PS.dTmin = dTminub;
			}
			if (Particle[p].PS.dTmin < dTminlb) {
				Particle[p].PS.dTmin = dTminlb;
			}
			double sumauxHU = 0;
			for (int u = 0; u <= CSPinch.Thuin.size() - 1; u++) {
				r1 = (double)rand() / (double)RAND_MAX;
				r2 = (double)rand() / (double)RAND_MAX;
				Particle[p].PS.VelHUFrac[u] = ((wmax - wmin) * exp(-PSOIter / PSOMaxIter) + wmin) * Particle[p].PS.VelHUFrac[u] + c1 * r1 * (ParticleBest[p].PS.HUFrac[u] - Particle[p].PS.HUFrac[u]) + c2 * r2 * (GlobalBest.PS.HUFrac[u] - Particle[p].PS.HUFrac[u]);
				Particle[p].PS.HUFrac[u] = Particle[p].PS.HUFrac[u] + Particle[p].PS.VelHUFrac[u];
				if (Particle[p].PS.HUFrac[u] > fracub) {
					Particle[p].PS.HUFrac[u] = fracub;
				}
				else if (Particle[p].PS.HUFrac[u] < fraclb + 0.001) {
					Particle[p].PS.HUFrac[u] = fraclb;
				}
				sumauxHU = sumauxHU + Particle[p].PS.HUFrac[u];
			}
			for (int u = 0; u <= CSPinch.Thuin.size() - 1; u++) {
				Particle[p].PS.HUFrac[u] = Particle[p].PS.HUFrac[u] / sumauxHU;
			}
			double sumauxCU = 0;
			for (int u = 0; u <= CSPinch.Tcuin.size() - 1; u++) {
				r1 = (double)rand() / (double)RAND_MAX;
				r2 = (double)rand() / (double)RAND_MAX;
				Particle[p].PS.VelCUFrac[u] = ((wmax - wmin) * exp(-PSOIter / PSOMaxIter) + wmin) * Particle[p].PS.VelCUFrac[u] + c1 * r1 * (ParticleBest[p].PS.CUFrac[u] - Particle[p].PS.CUFrac[u]) + c2 * r2 * (GlobalBest.PS.CUFrac[u] - Particle[p].PS.CUFrac[u]);
				Particle[p].PS.CUFrac[u] = Particle[p].PS.CUFrac[u] + Particle[p].PS.VelCUFrac[u];
				if (Particle[p].PS.CUFrac[u] > fracub) {
					Particle[p].PS.CUFrac[u] = fracub;
				}
				else if (Particle[p].PS.CUFrac[u] < fraclb + 0.001) {
					Particle[p].PS.CUFrac[u] = fraclb;
				}
				sumauxCU = sumauxCU + Particle[p].PS.CUFrac[u];
			}
			for (int u = 0; u <= CSPinch.Tcuin.size() - 1; u++) {
				Particle[p].PS.CUFrac[u] = Particle[p].PS.CUFrac[u] / sumauxCU;
			}

			//Sys(CaseStudy, Particle[p], 0);
			//BuildHICS(Particle[p], CaseStudy, CS, CSzero);
			PI = PIzero;
			Pinch(CSPinch, Particle[p].PS, PI, 0, start, today);
			Particle[p].TotalCosts = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts) + CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
			Particle[p].TPenCosts = Particle[p].TPenCosts + PI.TPenCosts;

			if (Particle[p].TotalCosts < ParticleBest[p].TotalCosts) {
				ParticleBest[p] = Particle[p];
			}
			if (Particle[p].TotalCosts < BestCostTemp) {
				BestCostInd = p;
				BestCostTemp = Particle[p].TotalCosts;
				improved = 1;
			}
		}
		if (improved == 1) {
			improved = 0;
			GlobalBest = ParticleBest[BestCostInd];
			//Sys(CaseStudy, Particle[p], 0);
			//BuildHICS(Particle[p], CaseStudy, CS, CSzero);
			PI = PIzero;
			Pinch(CSPinch, GlobalBest.PS, PI, 0, start, today);
			GlobalBest.TotalCosts = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts) + CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
			GlobalBest.TPenCosts = GlobalBest.TPenCosts + PI.TPenCosts;

			while (GlobalBest.TotalCosts > ParticleBest[BestCostInd].TotalCosts + 10000 && isnan(GlobalBest.TotalCosts) == 0) {
				GlobalBest = ParticleBest[BestCostInd];
				//Sys(CaseStudy, GlobalBest, 0);
				//BuildHICS(GlobalBest, CaseStudy, CS, CSzero);
				PI = PIzero;
				Pinch(CSPinch, GlobalBest.PS, PI, 0, start, today);
				GlobalBest.TotalCosts = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts) + CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
				GlobalBest.TPenCosts = GlobalBest.TPenCosts + PI.TPenCosts;
			}
		}
		//cout << "PSO Best ";
		//cout << GlobalBest.SysTACHI;
		//cout << "\t";
	}

	//Sys(CaseStudy, GlobalBest, 0);
	//BuildHICS(GlobalBest, CaseStudy, CS, CSzero);

	PI = PIzero;
	Pinch(CSPinch, GlobalBest.PS, PI, 0, start, today);
	GlobalBest.TotalCosts = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts) + CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
	GlobalBest.TPenCosts = GlobalBest.TPenCosts + PI.TPenCosts;

	while (GlobalBest.TotalCosts > ParticleBest[BestCostInd].TotalCosts + 10000 && isnan(GlobalBest.TotalCosts) == 0) {
		GlobalBest = ParticleBest[BestCostInd];
		//Sys(CaseStudy, GlobalBest, 0);
		//BuildHICS(GlobalBest, CaseStudy, CS, CSzero);

		PI = PIzero;
		Pinch(CSPinch, GlobalBest.PS, PI, 0, start, today);
		GlobalBest.TotalCosts = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts) + CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
		GlobalBest.TPenCosts = GlobalBest.TPenCosts + PI.TPenCosts;
	}
	BestSol = GlobalBest;

}

void CSA(double cT0, double cTf, double alpha, int cLmax, int Input, SolutionStruct InputSolution, SolutionStruct StructureSolution, PinchInterm PI, CaseStudyStruct CaseStudy, CaseStudyPinch CSPinch, CaseStudyPinch CSzero, SolutionStruct& BestCSol, double dTminub, double dTminlb, double fraclb, double fracub, double start, vector<int> today) {

	SolutionStruct NewCSol;
	SolutionStruct CurrentCSol;
	SolutionStruct GlobalBest;

	PinchInterm PIzero = PI;

	double BestCostTemp = +INFINITY;
	int improved = 0;
	int BestCostInd = 0;

	double VelFactor = 0.1;

	CurrentCSol = StructureSolution;
	NewCSol = StructureSolution;
	BestCSol = StructureSolution;

	if (Input == 1) {

		//Sys(CaseStudy, InputSolution, 0);

		//BuildHICS(InputSolution, CaseStudy, CS, CSzero);
		PI = PIzero;
		Pinch(CSPinch, InputSolution.PS, PI, 0, start, today);
		InputSolution.TotalCosts = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts) + CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
		InputSolution.TPenCosts = InputSolution.TPenCosts + PI.TPenCosts;

		NewCSol = InputSolution;
		CurrentCSol = InputSolution;
		BestCSol = InputSolution;
		/*
		int contt = 0;
		NewCSol.TPenCosts = 2.0;
		PI.TPenCosts = 1.0;
		while ((NewCSol.TPenCosts - PI.TPenCosts) > 0.1 && contt <= 500) {
			for (int k = 0; k <= NewCSol.Str.size() - 1; k++) {
				for (int i = 0; i <= NewCSol.Str[k].size() - 1; i++) {
					if (NewCSol.Str[k][i] == 1) {
						NewCSol.Col[k][i].q = 1;
						NewCSol.Col[k][i].Rfactor = Rfactorlb + (Rfactorub - Rfactorlb) * (double)rand() / (double)RAND_MAX;
						NewCSol.Col[k][i].P = Plb + (Pub - Plb) * (double)rand() / (double)RAND_MAX;
						NewCSol.Col[k][i].recLK = recLKlb + (recLKub - recLKlb) * (double)rand() / (double)RAND_MAX;
						NewCSol.Col[k][i].recHK = recHKlb + (recHKub - recHKlb) * (double)rand() / (double)RAND_MAX;

						NewCSol.Col[k][i].Velq = 0;
						NewCSol.Col[k][i].VelRfactor = VelFactor * (Rfactorub - Rfactorlb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
						NewCSol.Col[k][i].VelP = VelFactor * (Pub - Plb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
						NewCSol.Col[k][i].VelrecLK = VelFactor * (recLKub - recLKlb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
						NewCSol.Col[k][i].VelrecHK = VelFactor * (recHKub - recHKlb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
					}
				}
			}
			NewCSol.PS.dTmin = dTminlb + (dTminub - dTminlb) * (double)rand() / (double)RAND_MAX;
			//NewCSol.PS.VeldTmin = VelFactor * (dTminub - dTminlb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);


			double sumaux = 0;
			for (int u = 0; u <= CaseStudy.Thuin.size() - 1; u++) {
				NewCSol.PS.HUFrac[u] = fraclb + (fracub - fraclb) * (double)rand() / (double)RAND_MAX;
				sumaux = sumaux + NewCSol.PS.HUFrac[u];
			}
			for (int u = 0; u <= CaseStudy.Thuin.size() - 1; u++) {
				NewCSol.PS.HUFrac[u] = NewCSol.PS.HUFrac[u] / sumaux;
				NewCSol.PS.VelHUFrac[u] = VelFactor * (fracub - fraclb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
				NewCSol.PS.VarHUTempOut[u] = CaseStudy.Thuout[u];
				NewCSol.PS.VelVarHUTempOut[u] = 0;
			}
			sumaux = 0;
			for (int u = 0; u <= CaseStudy.Tcuin.size() - 1; u++) {
				NewCSol.PS.CUFrac[u] = fraclb + (fracub - fraclb) * (double)rand() / (double)RAND_MAX;
				sumaux = sumaux + NewCSol.PS.CUFrac[u];
			}
			for (int u = 0; u <= CaseStudy.Tcuin.size() - 1; u++) {
				NewCSol.PS.CUFrac[u] = NewCSol.PS.CUFrac[u] / sumaux;
				NewCSol.PS.VelCUFrac[u] = VelFactor * (fracub - fraclb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
				NewCSol.PS.VarCUTempOut[u] = CaseStudy.Tcuout[u];
				NewCSol.PS.VelVarCUTempOut[u] = 0;
			}

			Sys(CaseStudy, NewCSol, 0);
			BuildHICS(NewCSol, CaseStudy, CS, CSzero);
			PI = PIzero;
			Pinch(CS, NewCSol.PS, PI, 0, start, today);
			NewCSol.SysOCHI = OCHI(PI.HU, PI.CU, CS.HUcosts, CS.CUcosts, CaseStudy.hpery);
			NewCSol.SysCCHI = CCHI(PI.Nun, PI.intervals, PI.Areak, CS.B0, CS.C0, CS.beta0);
			NewCSol.TPenCosts = NewCSol.TPenCosts + PI.TPenCosts;
			NewCSol.SysTACHI = NewCSol.TPenCosts + NewCSol.SysColCC + NewCSol.SysOCHI + NewCSol.SysCCHI;
			//cout << NewCSol.SysTACHI; cout << "\n";

			contt++;
		}
		int aaaaaaa = 1;
		*/
	}
	else {

		//Inicializando vari�veis cont�nuas -- J� foram inicializadas no move do SA
		/*
		int contt = 0;
		NewCSol.TPenCosts = 2.0;
		PI.TPenCosts = 1.0;
		while ((NewCSol.TPenCosts - PI.TPenCosts) > 0.1 && contt <= 500) {
			for (int k = 0; k <= NewCSol.Str.size() - 1; k++) {
				for (int i = 0; i <= NewCSol.Str[k].size() - 1; i++) {
					if (NewCSol.Str[k][i] == 1) {
						NewCSol.Col[k][i].q = 1;
						NewCSol.Col[k][i].Rfactor = Rfactorlb + (Rfactorub - Rfactorlb) * (double)rand() / (double)RAND_MAX;
						NewCSol.Col[k][i].P = Plb + (Pub - Plb) * (double)rand() / (double)RAND_MAX;
						NewCSol.Col[k][i].recLK = recLKlb + (recLKub - recLKlb) * (double)rand() / (double)RAND_MAX;
						NewCSol.Col[k][i].recHK = recHKlb + (recHKub - recHKlb) * (double)rand() / (double)RAND_MAX;

						NewCSol.Col[k][i].Velq = 0;
						NewCSol.Col[k][i].VelRfactor = VelFactor * (Rfactorub - Rfactorlb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
						NewCSol.Col[k][i].VelP = VelFactor * (Pub - Plb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
						NewCSol.Col[k][i].VelrecLK = VelFactor * (recLKub - recLKlb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
						NewCSol.Col[k][i].VelrecHK = VelFactor * (recHKub - recHKlb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
					}
				}
			}
			NewCSol.PS.dTmin = dTminlb + (dTminub - dTminlb) * (double)rand() / (double)RAND_MAX;
			//NewCSol.PS.VeldTmin = VelFactor * (dTminub - dTminlb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);


			double sumaux = 0;
			for (int u = 0; u <= CaseStudy.Thuin.size() - 1; u++) {
				NewCSol.PS.HUFrac[u] = fraclb + (fracub - fraclb) * (double)rand() / (double)RAND_MAX;
				sumaux = sumaux + NewCSol.PS.HUFrac[u];
			}
			for (int u = 0; u <= CaseStudy.Thuin.size() - 1; u++) {
				NewCSol.PS.HUFrac[u] = NewCSol.PS.HUFrac[u] / sumaux;
				NewCSol.PS.VelHUFrac[u] = VelFactor * (fracub - fraclb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
				NewCSol.PS.VarHUTempOut[u] = CaseStudy.Thuout[u];
				NewCSol.PS.VelVarHUTempOut[u] = 0;
			}
			sumaux = 0;
			for (int u = 0; u <= CaseStudy.Tcuin.size() - 1; u++) {
				NewCSol.PS.CUFrac[u] = fraclb + (fracub - fraclb) * (double)rand() / (double)RAND_MAX;
				sumaux = sumaux + NewCSol.PS.CUFrac[u];
			}
			for (int u = 0; u <= CaseStudy.Tcuin.size() - 1; u++) {
				NewCSol.PS.CUFrac[u] = NewCSol.PS.CUFrac[u] / sumaux;
				NewCSol.PS.VelCUFrac[u] = VelFactor * (fracub - fraclb) * 2 * ((double)rand() / (double)RAND_MAX - 0.5);
				NewCSol.PS.VarCUTempOut[u] = CaseStudy.Tcuout[u];
				NewCSol.PS.VelVarCUTempOut[u] = 0;
			}

			Sys(CaseStudy, NewCSol, 0);
			BuildHICS(NewCSol, CaseStudy, CS, CSzero);
			PI = PIzero;
			Pinch(CS, NewCSol.PS, PI, 0, start, today);
			NewCSol.SysOCHI = OCHI(PI.HU, PI.CU, CS.HUcosts, CS.CUcosts, CaseStudy.hpery);
			NewCSol.SysCCHI = CCHI(PI.Nun, PI.intervals, PI.Areak, CS.B0, CS.C0, CS.beta0);
			NewCSol.TPenCosts = NewCSol.TPenCosts + PI.TPenCosts;
			NewCSol.SysTACHI = NewCSol.TPenCosts + NewCSol.SysColCC + NewCSol.SysOCHI + NewCSol.SysCCHI;
			//cout << NewCSol.SysTACHI; cout << "\n";

			contt++;
		}
		*/
		int aaaaaaa = 1;
	}


	CurrentCSol = NewCSol;

	double end = clock();
	double TotalTime2 = (double)(end - start) / CLOCKS_PER_SEC;
	printf_s("CSA ini %.2f (%.3f s)\t", CurrentCSol.TotalCosts, TotalTime2);

	double cT = cT0;
	while (cT > cTf) {
		double csavelfactor = (cT - cTf) / (cT0 - cTf);
		for (int l = 0; l <= cLmax; l++) {

			//Pinch move

			int sum1 = 1 + NewCSol.PS.HUFrac.size() + NewCSol.PS.CUFrac.size();
			int movetypeball = rand() % sum1;

			if (movetypeball == 0) {
				NewCSol.PS.dTmin = NewCSol.PS.dTmin + 0.1 * csavelfactor * (dTminub - dTminlb) * ((double)rand() / (double)RAND_MAX - 0.5);
				if (NewCSol.PS.dTmin > dTminub) {
					NewCSol.PS.dTmin = dTminub;
				}
				if (NewCSol.PS.dTmin < dTminlb) {
					NewCSol.PS.dTmin = dTminlb;
				}
			}
			else if (movetypeball > 0 && movetypeball <= (NewCSol.PS.HUFrac.size())) {
				int u = movetypeball - 1;
				NewCSol.PS.HUFrac[u] = NewCSol.PS.HUFrac[u] + 0.1 * csavelfactor * (fracub - fraclb) * ((double)rand() / (double)RAND_MAX - 0.5);
				if (NewCSol.PS.HUFrac[u] > fracub) {
					NewCSol.PS.HUFrac[u] = fracub;
				}
				else if (NewCSol.PS.HUFrac[u] < fraclb + 0.001) {
					NewCSol.PS.HUFrac[u] = fraclb;
				}
				double sumauxHU = 0;
				for (u = 0; u <= CSPinch.Thuin.size() - 1; u++) {
					sumauxHU = sumauxHU + NewCSol.PS.HUFrac[u];
				}
				for (u = 0; u <= CSPinch.Thuin.size() - 1; u++) {
					NewCSol.PS.HUFrac[u] = NewCSol.PS.HUFrac[u] / sumauxHU;
				}
			}
			else {
				int u = movetypeball - NewCSol.PS.HUFrac.size() - 1;
				NewCSol.PS.CUFrac[u] = NewCSol.PS.CUFrac[u] + 0.1 * csavelfactor * (fracub - fraclb) * ((double)rand() / (double)RAND_MAX - 0.5);
				if (NewCSol.PS.CUFrac[u] > fracub) {
					NewCSol.PS.CUFrac[u] = fracub;
				}
				else if (NewCSol.PS.CUFrac[u] < fraclb + 0.001) {
					NewCSol.PS.CUFrac[u] = fraclb;
				}
				double sumauxCU = 0;
				for (u = 0; u <= CSPinch.Tcuin.size() - 1; u++) {
					sumauxCU = sumauxCU + NewCSol.PS.CUFrac[u];
				}
				for (u = 0; u <= CSPinch.Tcuin.size() - 1; u++) {
					NewCSol.PS.CUFrac[u] = NewCSol.PS.CUFrac[u] / sumauxCU;
				}
			}


			//Sys(CaseStudy, InputSolution, 0);

			//BuildHICS(InputSolution, CaseStudy, CS, CSzero);
			PI = PIzero;
			Pinch(CSPinch, NewCSol.PS, PI, 0, start, today);
			NewCSol.TotalCosts = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts) + CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
			NewCSol.TPenCosts = NewCSol.TPenCosts + PI.TPenCosts;


			if (NewCSol.TotalCosts < CurrentCSol.TotalCosts) {
				CurrentCSol = NewCSol;
			}
			else {
				double SAPr = exp((CurrentCSol.TotalCosts - NewCSol.TotalCosts) / cT);
				double randomnum = (double)rand() / (double)RAND_MAX;
				if (randomnum < SAPr) {
					CurrentCSol = NewCSol;
				}
				else {
					NewCSol = CurrentCSol;
				}
			}

			if (NewCSol.TotalCosts < BestCSol.TotalCosts) {
				BestCSol = NewCSol;
				//cout << "CSA Best ";
				//cout << CurrentCSol.SysTACHI;
				//cout << "\n";
			}

		}
		cT = cT * alpha;
	}

	SolutionStruct BestSolTemp;
	BestSolTemp = BestCSol;
	if (BestCSol.TPenCosts == 0) {
		//Sys(CaseStudy, InputSolution, 0);

		//BuildHICS(InputSolution, CaseStudy, CS, CSzero);
		PI = PIzero;
		Pinch(CSPinch, NewCSol.PS, PI, 0, start, today);
		BestCSol.TotalCosts = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts) + CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
		BestCSol.TPenCosts = BestCSol.TPenCosts + PI.TPenCosts;


		while (BestCSol.TotalCosts > BestSolTemp.TotalCosts + 10000 && isnan(BestCSol.TotalCosts) == 0) {
			//Sys(CaseStudy, InputSolution, 0);

			//BuildHICS(InputSolution, CaseStudy, CS, CSzero);
			PI = PIzero;
			Pinch(CSPinch, NewCSol.PS, PI, 0, start, today);
			BestCSol.TotalCosts = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts) + CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
			BestCSol.TPenCosts = BestCSol.TPenCosts + PI.TPenCosts;
		}
	}
	end = clock();
	TotalTime2 = (double)(end - start) / CLOCKS_PER_SEC;
	printf_s("CSA Best %.2f (%.3f s)\t", BestCSol.TotalCosts, TotalTime2);

	int AAAAAAAAA = 1;

}


void BuildHENSyn0(int nstages, CaseStudyHENStruct& CaseStudyHEN, HENSynInterm& HSI, HENSolutionStruct& HENSolution) {
	int erasenullutil = 0;

	int nhot = 0; int ncold = 0;
	for (int i = 0; i <= (CaseStudyHEN.Streams.size()) - 1; i++) {
		if (CaseStudyHEN.Streams[i][1] >= CaseStudyHEN.Streams[i][2]) {
			nhot++;
		}
		else {
			ncold++;
		}
	}

	CaseStudyHEN.Thuin.resize(CaseStudyHEN.HUStreams.size());
	CaseStudyHEN.Thuout.resize(CaseStudyHEN.HUStreams.size());
	CaseStudyHEN.hhu.resize(CaseStudyHEN.HUStreams.size());
	CaseStudyHEN.Bh.resize(CaseStudyHEN.HUStreams.size());
	CaseStudyHEN.Ch.resize(CaseStudyHEN.HUStreams.size());
	CaseStudyHEN.betah.resize(CaseStudyHEN.HUStreams.size());

	CaseStudyHEN.Tcuin.resize(CaseStudyHEN.CUStreams.size());
	CaseStudyHEN.Tcuout.resize(CaseStudyHEN.CUStreams.size());
	CaseStudyHEN.hcu.resize(CaseStudyHEN.CUStreams.size());
	CaseStudyHEN.Bc.resize(CaseStudyHEN.CUStreams.size());
	CaseStudyHEN.Cc.resize(CaseStudyHEN.CUStreams.size());
	CaseStudyHEN.betac.resize(CaseStudyHEN.CUStreams.size());

	for (int conthu = 0; conthu <= CaseStudyHEN.HUStreams.size() - 1; conthu++) {
		CaseStudyHEN.Thuin[conthu] = (CaseStudyHEN.HUStreams[conthu][1]);
		CaseStudyHEN.Thuout[conthu] = (CaseStudyHEN.HUStreams[conthu][2]);
		CaseStudyHEN.hhu[conthu] = (CaseStudyHEN.HUStreams[conthu][4]);
		CaseStudyHEN.Bh[conthu] = (CaseStudyHEN.Bh[conthu]);
		CaseStudyHEN.Ch[conthu] = (CaseStudyHEN.Ch[conthu]);
		CaseStudyHEN.betah[conthu] = (CaseStudyHEN.betah[conthu]);
	}
	for (int contcu = 0; contcu <= CaseStudyHEN.CUStreams.size() - 1; contcu++) {
		CaseStudyHEN.Tcuin[contcu] = (CaseStudyHEN.CUStreams[contcu][1]);
		CaseStudyHEN.Tcuout[contcu] = (CaseStudyHEN.CUStreams[contcu][2]);
		CaseStudyHEN.hcu[contcu] = (CaseStudyHEN.CUStreams[contcu][4]);
		CaseStudyHEN.Bc[contcu] = (CaseStudyHEN.Bc[contcu]);
		CaseStudyHEN.Cc[contcu] = (CaseStudyHEN.Cc[contcu]);
		CaseStudyHEN.betac[contcu] = (CaseStudyHEN.betac[contcu]);
	}

	CaseStudyHEN.nhot = nhot;// +CaseStudyHEN.HUStreams.size();
	CaseStudyHEN.ncold = ncold;// +CaseStudyHEN.CUStreams.size();

	CaseStudyHEN.BB.resize(nhot + CaseStudyHEN.HUStreams.size(), vector<double>(ncold + CaseStudyHEN.CUStreams.size()));
	CaseStudyHEN.CC.resize(nhot + CaseStudyHEN.HUStreams.size(), vector<double>(ncold + CaseStudyHEN.CUStreams.size()));
	CaseStudyHEN.betaa.resize(nhot + CaseStudyHEN.HUStreams.size(), vector<double>(ncold + CaseStudyHEN.CUStreams.size()));

	if (CaseStudyHEN.specialHE == 0) {
		for (int i = 0; i < nhot; i++) {
			for (int j = 0; j < ncold; j++) {
				CaseStudyHEN.BB[i][j] = CaseStudyHEN.B[0];
				CaseStudyHEN.CC[i][j] = CaseStudyHEN.C[0];
				CaseStudyHEN.betaa[i][j] = CaseStudyHEN.beta[0];
			}
		}
		for (int i = nhot; i < nhot + CaseStudyHEN.HUStreams.size(); i++) {
			for (int j = 0; j < ncold; j++) {
				CaseStudyHEN.BB[i][j] = CaseStudyHEN.B[0];
				CaseStudyHEN.CC[i][j] = CaseStudyHEN.C[0];
				CaseStudyHEN.betaa[i][j] = CaseStudyHEN.beta[0];
			}
		}
		for (int i = 0; i < nhot; i++) {
			for (int j = ncold; j < ncold + CaseStudyHEN.CUStreams.size(); j++) {
				CaseStudyHEN.BB[i][j] = CaseStudyHEN.B[0];
				CaseStudyHEN.CC[i][j] = CaseStudyHEN.C[0];
				CaseStudyHEN.betaa[i][j] = CaseStudyHEN.beta[0];
			}
		}
	}

	CaseStudyHEN.Thin.resize(nhot + CaseStudyHEN.HUStreams.size());
	CaseStudyHEN.Thfinal.resize(nhot + CaseStudyHEN.HUStreams.size());
	CaseStudyHEN.CPh.resize(nhot + CaseStudyHEN.HUStreams.size());
	CaseStudyHEN.hh.resize(nhot + CaseStudyHEN.HUStreams.size());
	CaseStudyHEN.ishotutil.resize(nhot + CaseStudyHEN.HUStreams.size());

	CaseStudyHEN.Tcin.resize(ncold + CaseStudyHEN.CUStreams.size());
	CaseStudyHEN.Tcfinal.resize(ncold + CaseStudyHEN.CUStreams.size());
	CaseStudyHEN.CPc.resize(ncold + CaseStudyHEN.CUStreams.size());
	CaseStudyHEN.hc.resize(ncold + CaseStudyHEN.CUStreams.size());
	CaseStudyHEN.iscoldutil.resize(ncold + CaseStudyHEN.CUStreams.size());

	int iii = 0;
	int jjj = 0;
	for (int i = 0; i < CaseStudyHEN.Streams.size(); i++) {
		if (CaseStudyHEN.Streams[i][1] >= CaseStudyHEN.Streams[i][2]) {
			CaseStudyHEN.Thin[iii] = (CaseStudyHEN.Streams[i][1]);
			CaseStudyHEN.Thfinal[iii] = (CaseStudyHEN.Streams[i][2]);
			CaseStudyHEN.CPh[iii] = (CaseStudyHEN.Streams[i][3]);
			CaseStudyHEN.hh[iii] = (CaseStudyHEN.Streams[i][4]);
			CaseStudyHEN.ishotutil[iii] = (0);
			iii++;
		}
		else if (CaseStudyHEN.Streams[i][1] < CaseStudyHEN.Streams[i][2]) {
			CaseStudyHEN.Tcin[jjj] = (CaseStudyHEN.Streams[i][1]);
			CaseStudyHEN.Tcfinal[jjj] = (CaseStudyHEN.Streams[i][2]);
			CaseStudyHEN.CPc[jjj] = (CaseStudyHEN.Streams[i][3]);
			CaseStudyHEN.hc[jjj] = (CaseStudyHEN.Streams[i][4]);
			CaseStudyHEN.iscoldutil[jjj] = (0);
			jjj++;
		}
	}
	for (int i = 0; i < CaseStudyHEN.HUStreams.size(); i++) {
		CaseStudyHEN.Thin[iii] = (CaseStudyHEN.HUStreams[i][1]);
		CaseStudyHEN.Thfinal[iii] = (CaseStudyHEN.HUStreams[i][2]);
		CaseStudyHEN.CPh[iii] = (CaseStudyHEN.HUStreams[i][3]);
		CaseStudyHEN.hh[iii] = (CaseStudyHEN.HUStreams[i][4]);
		CaseStudyHEN.ishotutil[iii] = (1);
		iii++;
	}
	for (int i = 0; i < CaseStudyHEN.CUStreams.size(); i++) {
		CaseStudyHEN.Tcin[jjj] = (CaseStudyHEN.CUStreams[i][1]);
		CaseStudyHEN.Tcfinal[jjj] = (CaseStudyHEN.CUStreams[i][2]);
		CaseStudyHEN.CPc[jjj] = (CaseStudyHEN.CUStreams[i][3]);
		CaseStudyHEN.hc[jjj] = (CaseStudyHEN.CUStreams[i][4]);
		CaseStudyHEN.iscoldutil[jjj] = (1);
		jjj++;
	}

	CaseStudyHEN.Qmin = 0.0001;
	CaseStudyHEN.Frmin = 0.0001;
	//CaseStudyHEN.nhot = 0;
	//CaseStudyHEN.ncold = 0;
	CaseStudyHEN.nhu = CaseStudyHEN.Thuin.size();
	CaseStudyHEN.ncu = CaseStudyHEN.Tcuin.size();
	CaseStudyHEN.EMAT = 1.0;

	CaseStudyHEN.AllStreams = CaseStudyHEN.Streams;
	CaseStudyHEN.allnhot = 0;// CaseStudyHEN.nhot;
	CaseStudyHEN.allncold = 0;// CaseStudyHEN.ncold;
	//CaseStudyHEN.Streams.insert(CaseStudyHEN.Streams.begin() + CaseStudyHEN.Thin.size(), CaseStudyHEN.HUStreams.begin(), CaseStudyHEN.HUStreams.begin() + CaseStudyHEN.HUStreams.size());
	for (int i = 0; i < CaseStudyHEN.HUStreams.size(); i++) {
		CaseStudyHEN.AllStreams.push_back(CaseStudyHEN.HUStreams[i]);
	}
	for (int i = 0; i < CaseStudyHEN.CUStreams.size(); i++) {
		CaseStudyHEN.AllStreams.push_back(CaseStudyHEN.CUStreams[i]);
	}

	for (int i = 0; i <= (CaseStudyHEN.AllStreams.size()) - 1; i++) {
		if (CaseStudyHEN.AllStreams[i][1] >= CaseStudyHEN.AllStreams[i][2]) {
			CaseStudyHEN.allnhot++;
		}
		else {
			CaseStudyHEN.allncold++;
		}
	}

	CaseStudyHEN.AllHotStreams.resize(CaseStudyHEN.Thin.size());
	iii = 0;
	for (int i = 0; i <= (CaseStudyHEN.AllStreams.size()) - 1; i++) {
		if (CaseStudyHEN.AllStreams[i][1] >= CaseStudyHEN.AllStreams[i][2]) {
			CaseStudyHEN.AllHotStreams[iii] = (CaseStudyHEN.AllStreams[i]);
			iii++;
		}
	}

	//for (int i = 0; i <= CaseStudyHEN.HUStreams.size() - 1; i++) {
	//	CaseStudyHEN.AllHotStreams.push_back(CaseStudyHEN.HUStreams[i]);
	//	iii++;
	//}
	CaseStudyHEN.AllColdStreams.resize(CaseStudyHEN.Tcin.size());
	iii = 0;
	for (int i = 0; i <= (CaseStudyHEN.AllStreams.size()) - 1; i++) {
		if (CaseStudyHEN.AllStreams[i][1] < CaseStudyHEN.AllStreams[i][2]) {
			CaseStudyHEN.AllColdStreams[iii] = (CaseStudyHEN.AllStreams[i]);
			iii++;
		}
	}

	//for (int i = 0; i <= CaseStudyHEN.CUStreams.size() - 1; i++) {
	//	CaseStudyHEN.AllColdStreams.push_back(CaseStudyHEN.CUStreams[i]);
	//	iii++;
	//}

	//cout << "Hot Streams" << endl;
	CaseStudyHEN.Qh.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.Qhk.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.hh.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.CPh.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.Thin.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.Thfinal.resize(CaseStudyHEN.AllHotStreams.size());
	int x = 0;
	for (x = 0; x <= CaseStudyHEN.nhot + CaseStudyHEN.nhu - 1; x++) {
		CaseStudyHEN.Qh[x] = CaseStudyHEN.AllHotStreams[x][3] * (CaseStudyHEN.AllHotStreams[x][1] - CaseStudyHEN.AllHotStreams[x][2]);
		CaseStudyHEN.Qhk[x] = CaseStudyHEN.Qh[x];
		CaseStudyHEN.hh[x] = CaseStudyHEN.AllHotStreams[x][4];
		CaseStudyHEN.CPh[x] = CaseStudyHEN.AllHotStreams[x][3];
		CaseStudyHEN.Thin[x] = CaseStudyHEN.AllHotStreams[x][1];
		CaseStudyHEN.Thfinal[x] = CaseStudyHEN.AllHotStreams[x][2];
	}
	CaseStudyHEN.Qc.resize(CaseStudyHEN.AllColdStreams.size());
	CaseStudyHEN.Qck.resize(CaseStudyHEN.AllColdStreams.size());
	CaseStudyHEN.hc.resize(CaseStudyHEN.AllColdStreams.size());
	CaseStudyHEN.CPc.resize(CaseStudyHEN.AllColdStreams.size());
	CaseStudyHEN.Tcin.resize(CaseStudyHEN.AllColdStreams.size());
	CaseStudyHEN.Tcfinal.resize(CaseStudyHEN.AllColdStreams.size());
	for (x = 0; x <= CaseStudyHEN.AllColdStreams.size() - 1; x++) {
		CaseStudyHEN.Qc[x] = CaseStudyHEN.AllColdStreams[x][3] * (CaseStudyHEN.AllColdStreams[x][2] - CaseStudyHEN.AllColdStreams[x][1]);
		CaseStudyHEN.Qck[x] = CaseStudyHEN.Qc[x];
		CaseStudyHEN.hc[x] = CaseStudyHEN.AllColdStreams[x][4];
		CaseStudyHEN.CPc[x] = CaseStudyHEN.AllColdStreams[x][3];
		CaseStudyHEN.Tcin[x] = CaseStudyHEN.AllColdStreams[x][1];
		CaseStudyHEN.Tcfinal[x] = CaseStudyHEN.AllColdStreams[x][2];

		//cout << Qc[x-nhot] << endl;
	}
	CaseStudyHEN.U.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.Ucu.resize(CaseStudyHEN.AllHotStreams.size());
	for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
		CaseStudyHEN.U[i].resize(CaseStudyHEN.AllColdStreams.size());
		for (int contcu = 0; contcu <= CaseStudyHEN.ncu - 1; contcu++) {
			CaseStudyHEN.Ucu[i].push_back(1 / ((1 / CaseStudyHEN.hh[i]) + (1 / CaseStudyHEN.hcu[contcu])));
		}
		for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
			CaseStudyHEN.U[i][j] = 1 / ((1 / CaseStudyHEN.hh[i]) + (1 / CaseStudyHEN.hc[j]));
		};
	};
	CaseStudyHEN.Uhu.resize(CaseStudyHEN.AllColdStreams.size());
	for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
		for (int conthu = 0; conthu <= CaseStudyHEN.nhu - 1; conthu++) {
			CaseStudyHEN.Uhu[j].push_back(1 / ((1 / CaseStudyHEN.hc[j]) + (1 / CaseStudyHEN.hhu[conthu])));
		}
	};

	HSI.LMTDcu.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Areacu.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Qcu.resize(CaseStudyHEN.AllHotStreams.size());

	HSI.Thfinal0.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Thk.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Thout.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Tcout.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.LMTD.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Area.resize(CaseStudyHEN.AllHotStreams.size());
	for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
		HSI.Thk[i].resize(nstages);
		HSI.Thout[i].resize(CaseStudyHEN.AllColdStreams.size());
		HSI.Tcout[i].resize(CaseStudyHEN.AllColdStreams.size());
		HSI.LMTD[i].resize(CaseStudyHEN.AllColdStreams.size());
		HSI.Area[i].resize(CaseStudyHEN.AllColdStreams.size());
		for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
			HSI.Thout[i][j].resize(nstages);
			HSI.Tcout[i][j].resize(nstages);
			HSI.LMTD[i][j].resize(nstages);
			HSI.Area[i][j].resize(nstages);
		}
	}
	HSI.Tck.resize(CaseStudyHEN.AllColdStreams.size());
	HSI.Tcfinal0.resize(CaseStudyHEN.AllColdStreams.size());
	HSI.LMTDhu.resize(CaseStudyHEN.AllColdStreams.size());
	HSI.Areahu.resize(CaseStudyHEN.AllColdStreams.size());
	HSI.Qhu.resize(CaseStudyHEN.AllColdStreams.size());

	for (int i = 0; i <= CaseStudyHEN.AllColdStreams.size() - 1; i++) {
		HSI.Tck[i].resize(nstages);
	}

	HSI.TotalQhu.resize(CaseStudyHEN.nhu);
	HSI.TotalQcu.resize(CaseStudyHEN.ncu);

	HSI.TPenCosts = 0;
	HSI.AreaCosts = 0;
	HSI.UtilCosts = 0;

	HENSolution.z.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.Q.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.Fh.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.Fc.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.VelQ.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.VelFh.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.VelFc.resize(CaseStudyHEN.AllHotStreams.size());

	for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
		HENSolution.z[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.Q[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.Fh[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.Fc[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.VelQ[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.VelFh[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.VelFc[i].resize(CaseStudyHEN.AllColdStreams.size());
		for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
			HENSolution.z[i][j].resize(nstages);
			HENSolution.Q[i][j].resize(nstages);
			HENSolution.Fh[i][j].resize(nstages);
			HENSolution.Fc[i][j].resize(nstages);
			HENSolution.VelQ[i][j].resize(nstages);
			HENSolution.VelFh[i][j].resize(nstages);
			HENSolution.VelFc[i][j].resize(nstages);
		}
	}
	HENSolution.zcu.resize(CaseStudyHEN.AllHotStreams.size());
	for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
		HENSolution.zcu[i].resize(CaseStudyHEN.ncu);
	}
	HENSolution.zhu.resize(CaseStudyHEN.AllColdStreams.size());
	for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
		HENSolution.zhu[j].resize(CaseStudyHEN.nhu);
	}

	int maxtemphuind = 0;
	double maxtemphu = -1000;
	for (x = 0; x <= CaseStudyHEN.nhu - 1; x++) {
		if (CaseStudyHEN.Thuin[x] > maxtemphu) {
			maxtemphuind = x;
			maxtemphu = CaseStudyHEN.Thuin[x];
		}
	}
	for (int i = 0; i <= CaseStudyHEN.AllColdStreams.size() - 1; i++) {
		HENSolution.zhu[i][maxtemphuind] = 1;
	}

	int mintempcuind = 0;
	double mintempcu = 1000;
	for (x = 0; x <= CaseStudyHEN.ncu - 1; x++) {
		if (CaseStudyHEN.Tcuin[x] < mintempcu) {
			mintempcuind = x;
			mintempcu = CaseStudyHEN.Tcuin[x];
		}
	}
	for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
		HENSolution.zcu[i][mintempcuind] = 1;
	}

	CaseStudyHEN.freeendT.resize(CaseStudyHEN.AllColdStreams.size());

}

void BuildHENSyn(PinchSolution PS, int nstages, CaseStudyHENStruct& CaseStudyHEN, HENSynInterm& HSI, HENSolutionStruct& HENSolution, CaseStudyPinch CaseStudyPinch, PinchInterm PI) {
	int erasenullutil = 0;

	CaseStudyHEN.Qmin = 0.0001;
	CaseStudyHEN.Frmin = 0.0001;
	CaseStudyHEN.nhot = 0;
	CaseStudyHEN.ncold = 0;
	CaseStudyHEN.nhu = CaseStudyPinch.Thuin.size();
	CaseStudyHEN.ncu = CaseStudyPinch.Tcuin.size();
	CaseStudyHEN.EMAT = 1.0;
	CaseStudyHEN.Streams.resize(CaseStudyPinch.Tin.size());
	CaseStudyHEN.C.resize(CaseStudyPinch.Tin.size());
	CaseStudyHEN.B.resize(CaseStudyPinch.Tin.size());
	CaseStudyHEN.beta.resize(CaseStudyPinch.Tin.size());

	//CaseStudyHEN.AF = AF;

	for (int i = 0; i <= CaseStudyPinch.Tin.size() - 1; i++) {
		CaseStudyHEN.Streams[i].resize(5);
		CaseStudyHEN.Streams[i][0] = i + 1;
		CaseStudyHEN.Streams[i][1] = CaseStudyPinch.Tin[i];
		CaseStudyHEN.Streams[i][2] = CaseStudyPinch.Tout[i];
		CaseStudyHEN.Streams[i][3] = CaseStudyPinch.CP[i];
		CaseStudyHEN.Streams[i][4] = CaseStudyPinch.h[i];

		CaseStudyHEN.C[i] = CaseStudyPinch.C[i];
		CaseStudyHEN.B[i] = CaseStudyPinch.B[i];
		CaseStudyHEN.beta[i] = CaseStudyPinch.beta[i];
		if (CaseStudyPinch.Tin[i] >= CaseStudyPinch.Tout[i]) {
			CaseStudyHEN.nhot++;
		}
		if (CaseStudyPinch.Tin[i] < CaseStudyPinch.Tout[i]) {
			CaseStudyHEN.ncold++;
		}
	}

	//CaseStudyHEN.HUStreams.resize(CaseStudyPinch.Thuin.size());
	//CaseStudyHEN.HUCosts.resize(CaseStudyPinch.Thuin.size());
	//CaseStudyHEN.Ch.resize(CaseStudyPinch.Thuin.size());
	//CaseStudyHEN.Bh.resize(CaseStudyPinch.Thuin.size());
	//CaseStudyHEN.betah.resize(CaseStudyPinch.Thuin.size());
	CaseStudyHEN.nhu = 0;
	for (int i = 0; i <= CaseStudyPinch.Thuin.size() - 1; i++) {

		if ((PS.HUFrac[i] > 0.0001) || erasenullutil == 0) {
			CaseStudyHEN.nhu++;
			//CaseStudyHEN.HUStreams[i].resize(5);
			CaseStudyHEN.HUStreams.push_back({ 0,0,0,0,0 });
			CaseStudyHEN.HUCosts.push_back(0);
			CaseStudyHEN.Ch.push_back(0);
			CaseStudyHEN.Bh.push_back(0);
			CaseStudyHEN.betah.push_back(0);

			CaseStudyHEN.HUStreams[CaseStudyHEN.HUStreams.size() - 1][0] = i + 1;
			CaseStudyHEN.HUStreams[CaseStudyHEN.HUStreams.size() - 1][1] = CaseStudyPinch.Thuin[i];
			CaseStudyHEN.HUStreams[CaseStudyHEN.HUStreams.size() - 1][2] = CaseStudyPinch.Thuout[i];
			CaseStudyHEN.HUStreams[CaseStudyHEN.HUStreams.size() - 1][3] = 9999999;// CaseStudyPinch.CPhu[i];
			CaseStudyHEN.HUStreams[CaseStudyHEN.HUStreams.size() - 1][4] = CaseStudyPinch.hhu[i];
			CaseStudyHEN.Thuin.push_back(CaseStudyPinch.Thuin[i]);
			CaseStudyHEN.Thuout.push_back(CaseStudyPinch.Thuout[i]);
			CaseStudyHEN.HUCosts[CaseStudyHEN.HUStreams.size() - 1] = CaseStudyPinch.HUcosts[i];// 8500.0;
			CaseStudyHEN.Ch[CaseStudyHEN.HUStreams.size() - 1] = CaseStudyPinch.CHU[i];
			CaseStudyHEN.Bh[CaseStudyHEN.HUStreams.size() - 1] = CaseStudyPinch.BHU[i];
			CaseStudyHEN.betah[CaseStudyHEN.HUStreams.size() - 1] = CaseStudyPinch.betaHU[i];
		}

	}
	//CaseStudyHEN.CUStreams.resize(CaseStudyPinch.Tcuin.size());
	//CaseStudyHEN.CUCosts.resize(CaseStudyPinch.Tcuin.size());
	//CaseStudyHEN.Cc.resize(CaseStudyPinch.Tcuin.size());
	//CaseStudyHEN.Bc.resize(CaseStudyPinch.Tcuin.size());
	//CaseStudyHEN.betac.resize(CaseStudyPinch.Tcuin.size());
	CaseStudyHEN.ncu = 0;
	for (int i = 0; i <= CaseStudyPinch.Tcuin.size() - 1; i++) {

		if ((PS.CUFrac[i] > 0.0001) || erasenullutil == 0) {
			CaseStudyHEN.ncu++;
			//CaseStudyHEN.HUStreams[i].resize(5);
			CaseStudyHEN.CUStreams.push_back({ 0,0,0,0,0 });
			CaseStudyHEN.CUCosts.push_back(0);
			CaseStudyHEN.Cc.push_back(0);
			CaseStudyHEN.Bc.push_back(0);
			CaseStudyHEN.betac.push_back(0);

			CaseStudyHEN.CUStreams[CaseStudyHEN.CUStreams.size() - 1].resize(5);
			CaseStudyHEN.CUStreams[CaseStudyHEN.CUStreams.size() - 1][0] = i + 1;
			CaseStudyHEN.CUStreams[CaseStudyHEN.CUStreams.size() - 1][1] = CaseStudyPinch.Tcuin[i];
			CaseStudyHEN.CUStreams[CaseStudyHEN.CUStreams.size() - 1][2] = CaseStudyPinch.Tcuout[i];
			CaseStudyHEN.CUStreams[CaseStudyHEN.CUStreams.size() - 1][3] = 9999999;// CaseStudyPinch.CPcu[i];
			CaseStudyHEN.CUStreams[CaseStudyHEN.CUStreams.size() - 1][4] = CaseStudyPinch.hcu[i];
			CaseStudyHEN.Tcuin.push_back(CaseStudyPinch.Tcuin[i]);
			CaseStudyHEN.Tcuout.push_back(CaseStudyPinch.Tcuout[i]);
			CaseStudyHEN.CUCosts[CaseStudyHEN.CUStreams.size() - 1] = CaseStudyPinch.CUcosts[i];
			CaseStudyHEN.Cc[CaseStudyHEN.CUStreams.size() - 1] = CaseStudyPinch.CCU[i];
			CaseStudyHEN.Bc[CaseStudyHEN.CUStreams.size() - 1] = CaseStudyPinch.BCU[i];
			CaseStudyHEN.betac[CaseStudyHEN.CUStreams.size() - 1] = CaseStudyPinch.betaCU[i];
		}
	}

	int iii = 0;
	for (int i = 0; i <= (CaseStudyPinch.Tin.size()) - 1; i++) {
		if (CaseStudyHEN.Streams[i][1] >= CaseStudyHEN.Streams[i][2]) {
			CaseStudyHEN.AllHotStreams.push_back(CaseStudyHEN.Streams[i]);


			CaseStudyHEN.AllHotC.push_back(CaseStudyHEN.C[i]);
			CaseStudyHEN.AllHotB.push_back(CaseStudyHEN.B[i]);
			CaseStudyHEN.AllHotbeta.push_back(CaseStudyHEN.beta[i]);;

			iii++;
		}
	}

	CaseStudyHEN.ishotutil.resize(CaseStudyHEN.AllHotStreams.size());
	for (int i = 0; i <= CaseStudyHEN.HUStreams.size() - 1; i++) {
		CaseStudyHEN.AllHotStreams.push_back(CaseStudyHEN.HUStreams[i]);
		CaseStudyHEN.hhu.push_back(CaseStudyHEN.HUStreams[i][4]);
		CaseStudyHEN.ishotutil.push_back(i + 1);
		CaseStudyHEN.AllHotC.push_back(CaseStudyHEN.Ch[i]);
		CaseStudyHEN.AllHotB.push_back(CaseStudyHEN.Bh[i]);
		CaseStudyHEN.AllHotbeta.push_back(CaseStudyHEN.betah[i]);
		iii++;
	}
	iii = 0;
	for (int i = 0; i <= (CaseStudyPinch.Tin.size()) - 1; i++) {
		if (CaseStudyHEN.Streams[i][1] < CaseStudyHEN.Streams[i][2]) {
			CaseStudyHEN.AllColdStreams.push_back(CaseStudyHEN.Streams[i]);

			CaseStudyHEN.AllColdC.push_back(CaseStudyHEN.C[i]);
			CaseStudyHEN.AllColdB.push_back(CaseStudyHEN.B[i]);
			CaseStudyHEN.AllColdbeta.push_back(CaseStudyHEN.beta[i]);
			iii++;
		}
	}

	CaseStudyHEN.iscoldutil.resize(CaseStudyHEN.AllColdStreams.size());
	for (int i = 0; i <= CaseStudyHEN.CUStreams.size() - 1; i++) {
		CaseStudyHEN.AllColdStreams.push_back(CaseStudyHEN.CUStreams[i]);
		CaseStudyHEN.hcu.push_back(CaseStudyHEN.CUStreams[i][4]);
		CaseStudyHEN.iscoldutil.push_back(i + 1);
		CaseStudyHEN.AllColdC.push_back(CaseStudyHEN.Cc[i]);
		CaseStudyHEN.AllColdB.push_back(CaseStudyHEN.Bc[i]);
		CaseStudyHEN.AllColdbeta.push_back(CaseStudyHEN.betac[i]);
		iii++;
	}



	//cout << "Hot Streams" << endl;
	CaseStudyHEN.Qh.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.Qhk.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.hh.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.CPh.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.Thin.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.Thfinal.resize(CaseStudyHEN.AllHotStreams.size());
	int x = 0;
	for (x = 0; x <= CaseStudyHEN.nhot + CaseStudyHEN.nhu - 1; x++) {
		CaseStudyHEN.Qh[x] = CaseStudyHEN.AllHotStreams[x][3] * (CaseStudyHEN.AllHotStreams[x][1] - CaseStudyHEN.AllHotStreams[x][2]);
		CaseStudyHEN.Qhk[x] = CaseStudyHEN.Qh[x];
		CaseStudyHEN.hh[x] = CaseStudyHEN.AllHotStreams[x][4];
		CaseStudyHEN.CPh[x] = CaseStudyHEN.AllHotStreams[x][3];
		CaseStudyHEN.Thin[x] = CaseStudyHEN.AllHotStreams[x][1];
		CaseStudyHEN.Thfinal[x] = CaseStudyHEN.AllHotStreams[x][2];
	}
	CaseStudyHEN.Qc.resize(CaseStudyHEN.AllColdStreams.size());
	CaseStudyHEN.Qck.resize(CaseStudyHEN.AllColdStreams.size());
	CaseStudyHEN.hc.resize(CaseStudyHEN.AllColdStreams.size());
	CaseStudyHEN.CPc.resize(CaseStudyHEN.AllColdStreams.size());
	CaseStudyHEN.Tcin.resize(CaseStudyHEN.AllColdStreams.size());
	CaseStudyHEN.Tcfinal.resize(CaseStudyHEN.AllColdStreams.size());
	for (x = 0; x <= CaseStudyHEN.AllColdStreams.size() - 1; x++) {
		CaseStudyHEN.Qc[x] = CaseStudyHEN.AllColdStreams[x][3] * (CaseStudyHEN.AllColdStreams[x][2] - CaseStudyHEN.AllColdStreams[x][1]);
		CaseStudyHEN.Qck[x] = CaseStudyHEN.Qc[x];
		CaseStudyHEN.hc[x] = CaseStudyHEN.AllColdStreams[x][4];
		CaseStudyHEN.CPc[x] = CaseStudyHEN.AllColdStreams[x][3];
		CaseStudyHEN.Tcin[x] = CaseStudyHEN.AllColdStreams[x][1];
		CaseStudyHEN.Tcfinal[x] = CaseStudyHEN.AllColdStreams[x][2];

		//cout << Qc[x-nhot] << endl;
	}
	CaseStudyHEN.U.resize(CaseStudyHEN.AllHotStreams.size());
	CaseStudyHEN.Ucu.resize(CaseStudyHEN.AllHotStreams.size());
	for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
		CaseStudyHEN.U[i].resize(CaseStudyHEN.AllColdStreams.size());
		for (int contcu = 0; contcu <= CaseStudyHEN.ncu - 1; contcu++) {
			CaseStudyHEN.Ucu[i].push_back(1 / ((1 / CaseStudyHEN.hh[i]) + (1 / CaseStudyHEN.hcu[contcu])));
		}
		for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
			CaseStudyHEN.U[i][j] = 1 / ((1 / CaseStudyHEN.hh[i]) + (1 / CaseStudyHEN.hc[j]));
		};
	};
	CaseStudyHEN.Uhu.resize(CaseStudyHEN.AllColdStreams.size());
	for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
		for (int conthu = 0; conthu <= CaseStudyHEN.nhu - 1; conthu++) {
			CaseStudyHEN.Uhu[j].push_back(1 / ((1 / CaseStudyHEN.hc[j]) + (1 / CaseStudyHEN.hhu[conthu])));
		}
	};

	HSI.LMTDcu.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Areacu.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Qcu.resize(CaseStudyHEN.AllHotStreams.size());

	HSI.Thfinal0.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Thk.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Thout.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Tcout.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.LMTD.resize(CaseStudyHEN.AllHotStreams.size());
	HSI.Area.resize(CaseStudyHEN.AllHotStreams.size());
	for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
		HSI.Thk[i].resize(nstages);
		HSI.Thout[i].resize(CaseStudyHEN.AllColdStreams.size());
		HSI.Tcout[i].resize(CaseStudyHEN.AllColdStreams.size());
		HSI.LMTD[i].resize(CaseStudyHEN.AllColdStreams.size());
		HSI.Area[i].resize(CaseStudyHEN.AllColdStreams.size());
		for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
			HSI.Thout[i][j].resize(nstages);
			HSI.Tcout[i][j].resize(nstages);
			HSI.LMTD[i][j].resize(nstages);
			HSI.Area[i][j].resize(nstages);
		}
	}
	HSI.Tck.resize(CaseStudyHEN.AllColdStreams.size());
	HSI.Tcfinal0.resize(CaseStudyHEN.AllColdStreams.size());
	HSI.LMTDhu.resize(CaseStudyHEN.AllColdStreams.size());
	HSI.Areahu.resize(CaseStudyHEN.AllColdStreams.size());
	HSI.Qhu.resize(CaseStudyHEN.AllColdStreams.size());

	for (int i = 0; i <= CaseStudyHEN.AllColdStreams.size() - 1; i++) {
		HSI.Tck[i].resize(nstages);
	}

	HSI.TotalQhu.resize(CaseStudyHEN.nhu);
	HSI.TotalQcu.resize(CaseStudyHEN.ncu);

	HSI.TPenCosts = 0;
	HSI.AreaCosts = 0;
	HSI.UtilCosts = 0;

	HENSolution.z.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.Q.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.Fh.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.Fc.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.VelQ.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.VelFh.resize(CaseStudyHEN.AllHotStreams.size());
	HENSolution.VelFc.resize(CaseStudyHEN.AllHotStreams.size());

	for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
		HENSolution.z[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.Q[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.Fh[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.Fc[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.VelQ[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.VelFh[i].resize(CaseStudyHEN.AllColdStreams.size());
		HENSolution.VelFc[i].resize(CaseStudyHEN.AllColdStreams.size());
		for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
			HENSolution.z[i][j].resize(nstages);
			HENSolution.Q[i][j].resize(nstages);
			HENSolution.Fh[i][j].resize(nstages);
			HENSolution.Fc[i][j].resize(nstages);
			HENSolution.VelQ[i][j].resize(nstages);
			HENSolution.VelFh[i][j].resize(nstages);
			HENSolution.VelFc[i][j].resize(nstages);
		}
	}
	HENSolution.zcu.resize(CaseStudyHEN.AllHotStreams.size());
	for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
		HENSolution.zcu[i].resize(CaseStudyHEN.ncu);
	}
	HENSolution.zhu.resize(CaseStudyHEN.AllColdStreams.size());
	for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
		HENSolution.zhu[j].resize(CaseStudyHEN.nhu);
	}

	int maxtemphuind = 0;
	double maxtemphu = -1000;
	for (x = 0; x <= CaseStudyHEN.nhu - 1; x++) {
		if (CaseStudyHEN.Thuin[x] > maxtemphu) {
			maxtemphuind = x;
			maxtemphu = CaseStudyHEN.Thuin[x];
		}
	}
	for (int i = 0; i <= CaseStudyHEN.AllColdStreams.size() - 1; i++) {
		HENSolution.zhu[i][maxtemphuind] = 1;
	}

	int mintempcuind = 0;
	double mintempcu = 1000;
	for (x = 0; x <= CaseStudyHEN.ncu - 1; x++) {
		if (CaseStudyHEN.Tcuin[x] < mintempcu) {
			mintempcuind = x;
			mintempcu = CaseStudyHEN.Tcuin[x];
		}
	}
	for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
		HENSolution.zcu[i][mintempcuind] = 1;
	}

	CaseStudyHEN.freeendT.resize(CaseStudyHEN.AllColdStreams.size());

}

void BuildSpagCases(CaseStudyPinch CS, PinchInterm PI, CaseStudyHENStruct CaseStudyHEN, vector<vector<vector<double>>>& Streams2) {
	int nhotori = CaseStudyHEN.nhot + CaseStudyHEN.nhu;
	int ncoldori = CaseStudyHEN.ncold + CaseStudyHEN.ncu;
	int nhot = nhotori;
	int ncold = ncoldori;
	//nstages = nstagesori;
	int nstrori = CaseStudyHEN.nhot + CaseStudyHEN.nhu + CaseStudyHEN.ncold + CaseStudyHEN.ncu;
	int nstr = nstrori;

	int nhotind = nhot - 1;
	int ncoldind = ncold - 1;

	double sumheath = 0;
	double sumheatc = 0;

	int attcont = 0;
	int TotalAreaIntervals = PI.intervals + 2;
	vector<int> Forbidh; Forbidh.resize(CaseStudyHEN.nhot + CaseStudyHEN.nhu);
	vector<int> Forbidc; Forbidc.resize(CaseStudyHEN.ncold + CaseStudyHEN.ncu);


	vector<double> TableThin2;
	TableThin2 = PI.TableThin2;
	vector<double> TableThout2;
	TableThout2 = PI.TableThout2;
	vector<double> Tablehh;
	Tablehh = PI.Tablehh;
	vector<double> Tablehc;
	Tablehc = PI.Tablehc;

	vector<double> TableTcin2;
	TableTcin2 = PI.TableTcin2;
	vector<double> TableTcout2;
	TableTcout2 = PI.TableTcout2;
	vector<double> TableCPh;
	TableCPh = PI.TableCPh;
	vector<double> TableCPc;
	TableCPc = PI.TableCPc;

	int ii = 0;
	int jj = 0;
	for (int i = CaseStudyHEN.nhot; i <= CaseStudyHEN.nhot + CaseStudyHEN.nhu - 1; i++) {
		TableThin2[i] = CS.Thuin[ii];
		TableThout2[i] = CS.Thuout[ii];
		TableCPh[i] = PI.HU[ii] / (CS.Thuin[ii] - CS.Thuout[ii]);
		Tablehh[i] = CS.hhu[ii];
		ii++;
	}
	for (int i = CaseStudyHEN.ncold; i <= CaseStudyHEN.ncold + CaseStudyHEN.ncu - 1; i++) {
		TableTcin2[i] = CS.Tcuin[jj];
		TableTcout2[i] = CS.Tcuout[jj];
		TableCPc[i] = PI.CU[jj] / (CS.Tcuout[jj] - CS.Tcuin[jj]);//CS.CPcu[jj];
		Tablehc[i] = CS.hcu[jj];
		jj++;
	}

	sumheath = 0;
	sumheatc = 0;

	Forbidh.clear(); Forbidc.clear();
	Forbidh.resize(CaseStudyHEN.nhot + CaseStudyHEN.nhu); Forbidc.resize(CaseStudyHEN.ncold + CaseStudyHEN.ncu);
	int conth = 0;
	for (int cont = 0; cont <= TotalAreaIntervals - 2; cont++) {
		Streams2.resize(cont + 1);
		for (conth = 0; conth <= nhot - 1; conth++) {
			Streams2[cont].resize(conth + 1); Streams2[cont][conth].resize(5);
			if (((TableThin2[conth] >= PI.IntervalTable[cont][1]) && (TableThout2[conth] <= PI.IntervalTable[cont + 1][1]))) {// && (IntervalTable[cont][0] - IntervalTable[cont + 1][0] > 0)) {					
				Streams2[cont][conth][0] = conth + 1;
				Streams2[cont][conth][1] = PI.IntervalTable[cont][1];
				Streams2[cont][conth][2] = PI.IntervalTable[cont + 1][1];
				Streams2[cont][conth][3] = TableCPh[conth];
				Streams2[cont][conth][4] = Tablehh[conth];
			}
			else {
				Streams2[cont][conth][0] = conth + 1;
				Streams2[cont][conth][1] = PI.IntervalTable[cont][1];
				Streams2[cont][conth][2] = PI.IntervalTable[cont + 1][1];
				Streams2[cont][conth][3] = 0;
				Streams2[cont][conth][4] = Tablehh[conth];
			}
		}
	}



	for (int cont = 0; cont <= TotalAreaIntervals - 2; cont++) {
		for (int contc = 0; contc <= ncold - 1; contc++) {
			Streams2[cont].resize(contc + conth + 1); Streams2[cont][contc + conth].resize(5);
			if (((TableTcout2[contc] >= PI.IntervalTable[cont][2]) && (TableTcin2[contc] <= PI.IntervalTable[cont + 1][2]))) {// && (IntervalTable[cont][0] - IntervalTable[cont + 1][0] > 0)) {
				Streams2[cont][contc + conth][0] = contc + conth + 1;
				Streams2[cont][contc + conth][1] = PI.IntervalTable[cont + 1][2];
				Streams2[cont][contc + conth][2] = PI.IntervalTable[cont][2];
				Streams2[cont][contc + conth][3] = TableCPc[contc];
				Streams2[cont][contc + conth][4] = Tablehc[contc];
			}
			else {
				Streams2[cont][contc + conth][0] = contc + conth + 1;
				Streams2[cont][contc + conth][1] = PI.IntervalTable[cont + 1][2];
				Streams2[cont][contc + conth][2] = PI.IntervalTable[cont][2];
				Streams2[cont][contc + conth][3] = 0;
				Streams2[cont][contc + conth][4] = Tablehc[contc];
			}
		}
	}



	int aaa = 0;

}

void SWS(int spaghetti, int nstages, CaseStudyHENStruct CaseStudyHEN, HENSolutionStruct& HENSolution, HENSynInterm HSI, int print) {
	double Qmin = CaseStudyHEN.Qmin;
	double Frmin = CaseStudyHEN.Frmin;
	int k = 0;
	int i = 0;
	int j = 0;
	int nstagesind = nstages - 1;
	int nhotind = CaseStudyHEN.nhot + CaseStudyHEN.nhu - 1;
	int ncoldind = CaseStudyHEN.ncold + CaseStudyHEN.ncu - 1;
	vector<vector<double>> sumQThk; sumQThk.resize(nhotind + 1);
	for (i = 0; i <= nhotind; i++) {
		sumQThk[i].resize(nstages);
	}
	vector<vector<double>> sumQTck; sumQTck.resize(ncoldind + 1);
	for (i = 0; i <= ncoldind; i++) {
		sumQTck[i].resize(nstages);
	}
	HSI.TPenCosts = 0;
	int p = 0;
	for (k = 0; k <= nstagesind; k++) {
		for (i = 0; i <= nhotind; i++) {
			sumQThk[i][k] = 0;
		}
	}
	for (k = 0; k <= nstagesind; k++) {
		for (j = 0; j <= ncoldind; j++) {
			sumQTck[j][k] = 0;
		}
	}
	//=== Objective Function Calculation =============================

	double UtilCosts = 0;
	double sumqstream = 0;
	for (i = 0; i <= nhotind; i++) {
		sumqstream = 0;
		if (CaseStudyHEN.ishotutil[i] > 0.0) {
			for (k = 0; k <= nstages - 1; k++) {
				for (j = 0; j <= ncoldind; j++) {
					if (HENSolution.Q[i][j][k] > Qmin && HENSolution.Fh[i][j][k] > Frmin && HENSolution.Fc[i][j][k] > Frmin) {
						sumqstream = sumqstream + HENSolution.Q[i][j][k];
					}
				}
			}
			UtilCosts = UtilCosts + CaseStudyHEN.HUCosts[CaseStudyHEN.ishotutil[i] - 1] * sumqstream;
			CaseStudyHEN.CPh[i] = sumqstream / (CaseStudyHEN.Thin[i] - CaseStudyHEN.Thfinal[i]);
		}
	}
	for (j = 0; j <= ncoldind; j++) {
		sumqstream = 0;
		if (CaseStudyHEN.iscoldutil[j] > 0.0) {
			for (k = 0; k <= nstages - 1; k++) {
				for (i = 0; i <= nhotind; i++) {
					if (HENSolution.Q[i][j][k] > Qmin && HENSolution.Fh[i][j][k] > Frmin && HENSolution.Fc[i][j][k] > Frmin) {
						sumqstream = sumqstream + HENSolution.Q[i][j][k];
					}
				}
			}
			UtilCosts = UtilCosts + CaseStudyHEN.CUCosts[CaseStudyHEN.iscoldutil[j] - 1] * sumqstream;
			CaseStudyHEN.CPc[j] = sumqstream / (CaseStudyHEN.Tcfinal[j] - CaseStudyHEN.Tcin[j]);
		}
	}

	for (i = 0; i <= nhotind; i++) {
		HSI.Thk[i][0] = CaseStudyHEN.Thin[i];
		for (k = 0; k <= nstages - 1; k++) {
			for (j = 0; j <= ncoldind; j++) {
				if (HENSolution.Q[i][j][k] > Qmin && HENSolution.Fh[i][j][k] > Frmin && HENSolution.Fc[i][j][k] > Frmin) {
					HSI.Thout[i][j][k] = HSI.Thk[i][k] - HENSolution.Q[i][j][k] / (HENSolution.Fh[i][j][k] * CaseStudyHEN.CPh[i]);
					//cout << i << j << k << " " << Thk[i][k] << " - " << Particle[p].Q[i][j][k] << "/" << "(" << Particle[p].Fh[i][j][k] << "*" << CPh[i] << ")" << " = " << Thout[i][j][k] << endl;
				};
			};
			if (k < nstagesind) {
				for (int jj = 0; jj <= ncoldind; jj++) {
					if (HENSolution.Q[i][jj][k] > Qmin && HENSolution.Fh[i][jj][k] > Frmin && HENSolution.Fc[i][jj][k] > Frmin) {
						sumQThk[i][k] = sumQThk[i][k] + HENSolution.Q[i][jj][k];
					}
				};
				HSI.Thk[i][k + 1] = HSI.Thk[i][k] - sumQThk[i][k] / CaseStudyHEN.CPh[i];
			}
			else {
				for (int jj = 0; jj <= ncoldind; jj++) {
					if (HENSolution.Q[i][jj][k] > Qmin && HENSolution.Fh[i][jj][k] > Frmin && HENSolution.Fc[i][jj][k] > Frmin) {
						sumQThk[i][k] = sumQThk[i][k] + HENSolution.Q[i][jj][k];
					}
				};
				HSI.Thfinal0[i] = HSI.Thk[i][k] - sumQThk[i][k] / CaseStudyHEN.CPh[i];
			};
		};
	};
	//cout << "Cold Temperatures" << endl;
	for (j = ncoldind; j >= 0; j--) {
		HSI.Tck[j][nstagesind] = CaseStudyHEN.Tcin[j];
		for (k = nstagesind; k >= 0; k--) {
			for (i = nhotind; i >= 0; i--) {
				if (HENSolution.Q[i][j][k] > Qmin && HENSolution.Fh[i][j][k] > Frmin && HENSolution.Fc[i][j][k] > Frmin) {
					HSI.Tcout[i][j][k] = HSI.Tck[j][k] + HENSolution.Q[i][j][k] / (HENSolution.Fc[i][j][k] * CaseStudyHEN.CPc[j]);
					//cout << i << j << k << " " << Tck[j][k] << " + " << Particle[p].Q[i][j][k] << "/" << "(" << Particle[p].Fc[i][j][k] << "*" << CPc[j] << ")" << " = " << Tcout[i][j][k];
					if (HSI.Thk[i][k] - HSI.Tcout[i][j][k] < CaseStudyHEN.EMAT || HSI.Thout[i][j][k] - HSI.Tck[j][k] < CaseStudyHEN.EMAT) {
						if (HSI.Thk[i][k] - HSI.Tcout[i][j][k] < CaseStudyHEN.EMAT) {
							HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.taCtroc + CaseStudyHEN.tbCtroc * (CaseStudyHEN.EMAT - (HSI.Thk[i][k] - HSI.Tcout[i][j][k]));
						}
						if (HSI.Thout[i][j][k] - HSI.Tck[j][k] < CaseStudyHEN.EMAT) {
							HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.taCtroc + CaseStudyHEN.tbCtroc * (CaseStudyHEN.EMAT - (HSI.Thout[i][j][k] - HSI.Tck[j][k]));
						}
					}
					else {
						//LMTD and Areas Calculation (((Thinl-Tcoutl).*(Thoutl-Tcinl).*((Thinl-Tcoutl)+(Thoutl-Tcinl)))/2).^(1/3);
						//LMTD[i][j][k] = pow((((Thk[i][k] - Tcout[i][j][k])*(Thout[i][j][k] - Tck[j][k])*((Thk[i][k] - Tcout[i][j][k]) + (Thout[i][j][k] - Tck[j][k]))) / 2), 0.3333333333333333);

						if (abs((HSI.Thk[i][k] - HSI.Tcout[i][j][k]) - (HSI.Thout[i][j][k] - HSI.Tck[j][k])) < 0.0001) {
							HSI.LMTD[i][j][k] = (HSI.Thk[i][k] - HSI.Tcout[i][j][k]);
						}
						else {
							HSI.LMTD[i][j][k] = ((HSI.Thk[i][k] - HSI.Tcout[i][j][k]) - (HSI.Thout[i][j][k] - HSI.Tck[j][k])) / log((HSI.Thk[i][k] - HSI.Tcout[i][j][k]) / (HSI.Thout[i][j][k] - HSI.Tck[j][k]));
						}

						//cout << " LMDT = " << LMTD[i][j][k];
						HSI.Area[i][j][k] = HENSolution.Q[i][j][k] / (CaseStudyHEN.U[i][j] * HSI.LMTD[i][j][k]);
						if (HSI.Area[i][j][k] > 0 && HSI.Area[i][j][k] < CaseStudyHEN.minarea) {
							HSI.TPenCosts = HSI.TPenCosts + 100 * CaseStudyHEN.taCtroc + CaseStudyHEN.taCtroc * pow((HSI.Area[i][j][k] - CaseStudyHEN.minarea), 2);
						}
						//cout << " Area = " << Area[i][j][k] << endl;
						HSI.AreaCosts = HSI.AreaCosts + (CaseStudyHEN.BB[i][j] + CaseStudyHEN.CC[i][j] * pow(HSI.Area[i][j][k], CaseStudyHEN.betaa[i][j]));
					}


				};
			};
			if (k > 0) {
				for (int ii = nhotind; ii >= 0; ii--) {
					if (HENSolution.Q[ii][j][k] > Qmin && HENSolution.Fh[ii][j][k] > Frmin && HENSolution.Fc[ii][j][k] > Frmin) {
						sumQTck[j][k] = sumQTck[j][k] + HENSolution.Q[ii][j][k];
					}
				};
				HSI.Tck[j][k - 1] = HSI.Tck[j][k] + sumQTck[j][k] / CaseStudyHEN.CPc[j];
			}
			else {
				for (int ii = nhotind; ii >= 0; ii--) {
					if (HENSolution.Q[ii][j][k] > Qmin && HENSolution.Fh[ii][j][k] > Frmin && HENSolution.Fc[ii][j][k] > Frmin) {
						sumQTck[j][k] = sumQTck[j][k] + HENSolution.Q[ii][j][k];
					}
				};
				HSI.Tcfinal0[j] = HSI.Tck[j][k] + sumQTck[j][k] / CaseStudyHEN.CPc[j];
			};
		};
	};
	//Areas for Heaters/Coolers
	for (i = 0; i <= nhotind; i++) {
		if (CaseStudyHEN.ishotutil[i] == 0) {
			HSI.Qcu[i] = CaseStudyHEN.CPh[i] * (HSI.Thfinal0[i] - CaseStudyHEN.Thfinal[i]);
			if (spaghetti == 1 && isnan(HSI.Qcu[i]) == 0) {
				UtilCosts = UtilCosts + 50000 * abs(HSI.Qcu[i]);
			}
			if (HSI.Qcu[i] < 0) {
				HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.qaCtroc + CaseStudyHEN.qbCtroc * pow((HSI.Qcu[i]), 2);
			}
			for (int contcu = 0; contcu <= CaseStudyHEN.ncu - 1; contcu++) {
				if (HENSolution.zcu[i][contcu] == 1) {
					if (HSI.Qcu[i] > Qmin) { //ADICIONAR OS ZCU E ZHU AQUI!!
						if (HSI.Thfinal0[i] - CaseStudyHEN.Tcuout[contcu] < CaseStudyHEN.EMAT || CaseStudyHEN.Thfinal[i] - CaseStudyHEN.Tcuin[contcu] < CaseStudyHEN.EMAT) {
							if (HSI.Thfinal0[i] - CaseStudyHEN.Tcuout[contcu] < CaseStudyHEN.EMAT) {
								if (spaghetti == 0)
									HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.taCcu + CaseStudyHEN.tbCcu * (CaseStudyHEN.EMAT - (HSI.Thfinal0[i] - CaseStudyHEN.Tcuout[contcu]));
							}
							if (CaseStudyHEN.Thfinal[i] - CaseStudyHEN.Tcuin[contcu] < CaseStudyHEN.EMAT) {
								if (spaghetti == 0)
									HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.taCcu + CaseStudyHEN.tbCcu * (CaseStudyHEN.EMAT - (CaseStudyHEN.Thfinal[i] - CaseStudyHEN.Tcuin[contcu]));
							}
						}
						else {
							HSI.TotalQcu[contcu] = HSI.TotalQcu[contcu] + HSI.Qcu[i];
							UtilCosts = UtilCosts + CaseStudyHEN.CUCosts[contcu] * HSI.Qcu[i];
							//HSI.LMTDcu[i] = pow((((HSI.Thfinal0[i] - CaseStudyHEN.Tcuout)*(CaseStudyHEN.Thfinal[i] - CaseStudyHEN.Tcuin)*((HSI.Thfinal0[i] - CaseStudyHEN.Tcuout) + (CaseStudyHEN.Thfinal[i] - CaseStudyHEN.Tcuin))) / 2), 0.3333333333333333);
							if (abs((HSI.Thfinal0[i] - CaseStudyHEN.Tcuout[contcu]) - (CaseStudyHEN.Thfinal[i] - CaseStudyHEN.Tcuin[contcu])) < 0.0001) {
								HSI.LMTDcu[i] = (HSI.Thfinal0[i] - CaseStudyHEN.Tcuout[contcu]);
							}
							else {
								HSI.LMTDcu[i] = ((HSI.Thfinal0[i] - CaseStudyHEN.Tcuout[contcu]) - (CaseStudyHEN.Thfinal[i] - CaseStudyHEN.Tcuin[contcu])) / log((HSI.Thfinal0[i] - CaseStudyHEN.Tcuout[contcu]) / (CaseStudyHEN.Thfinal[i] - CaseStudyHEN.Tcuin[contcu]));
							}
							HSI.Areacu[i] = HSI.Qcu[i] / (CaseStudyHEN.Ucu[i][contcu] * HSI.LMTDcu[i]);
							if (HSI.Areacu[i] > 0 && HSI.Areacu[i] < CaseStudyHEN.minarea) {
								HSI.TPenCosts = HSI.TPenCosts + 100 * CaseStudyHEN.taCtroc + CaseStudyHEN.taCtroc * pow((HSI.Areacu[i] - CaseStudyHEN.minarea), 2);
							}
							if (HSI.Areacu[i] < 0.0) {
								HSI.TPenCosts = HSI.TPenCosts + 100 * CaseStudyHEN.taCtroc + CaseStudyHEN.taCtroc * pow((HSI.Areacu[i]), 2);
							}
							else {
								HSI.AreaCosts = HSI.AreaCosts + (CaseStudyHEN.BB[i][(CaseStudyHEN.ncold - 1) + contcu] + CaseStudyHEN.CC[i][(CaseStudyHEN.ncold - 1) + contcu] * pow(HSI.Areacu[i], CaseStudyHEN.betaa[i][(CaseStudyHEN.ncold - 1) + contcu]));
							}
						}
					}
				}
			}
		}

	}
	for (j = 0; j <= ncoldind; j++) {
		if (CaseStudyHEN.iscoldutil[j] == 0) {
			if (CaseStudyHEN.freeendT[j] == 1) {
				HSI.Qhu[j] = 0;
			}
			else {
				HSI.Qhu[j] = CaseStudyHEN.CPc[j] * (CaseStudyHEN.Tcfinal[j] - HSI.Tcfinal0[j]);
				if (spaghetti == 1 && isnan(HSI.Qhu[j]) == 0) {
					UtilCosts = UtilCosts + 50000 * abs(HSI.Qhu[j]);
				}
			}

			if (HSI.Qhu[j] < 0) {
				HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.qaCtroc + CaseStudyHEN.qbCtroc * pow((HSI.Qhu[j]), 2);
			}
			for (int conthu = 0; conthu <= CaseStudyHEN.nhu - 1; conthu++) {
				if (HENSolution.zhu[j][conthu] == 1) {
					if (HSI.Qhu[j] > Qmin) {
						if (CaseStudyHEN.Thuin[conthu] - CaseStudyHEN.Tcfinal[j] < CaseStudyHEN.EMAT || CaseStudyHEN.Thuout[conthu] - HSI.Tcfinal0[j] < CaseStudyHEN.EMAT) {
							if (CaseStudyHEN.Thuin[conthu] - CaseStudyHEN.Tcfinal[j] < CaseStudyHEN.EMAT) {
								if (spaghetti == 0)
									HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.taChu + CaseStudyHEN.tbChu * (CaseStudyHEN.EMAT - (CaseStudyHEN.Thuin[conthu] - CaseStudyHEN.Tcfinal[j]));
							}
							if (CaseStudyHEN.Thuout[conthu] - HSI.Tcfinal0[j] < CaseStudyHEN.EMAT) {
								if (spaghetti == 0)
									HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.taChu + CaseStudyHEN.taChu * (CaseStudyHEN.EMAT - (CaseStudyHEN.Thuout[conthu] - HSI.Tcfinal0[j]));
							}
						}

						else {
							HSI.TotalQhu[conthu] = HSI.TotalQhu[conthu] + HSI.Qhu[j];
							UtilCosts = UtilCosts + CaseStudyHEN.HUCosts[conthu] * HSI.Qhu[j];
							//HSI.LMTDhu[j] = pow((((CaseStudyHEN.Thuin - CaseStudyHEN.Tcfinal[j])*(CaseStudyHEN.Thuout - HSI.Tcfinal0[j])*((CaseStudyHEN.Thuin - CaseStudyHEN.Tcfinal[j]) + (CaseStudyHEN.Thuout - HSI.Tcfinal0[j]))) / 2), 0.3333333333333333);
							if (abs((CaseStudyHEN.Thuin[conthu] - CaseStudyHEN.Tcfinal[j]) - (CaseStudyHEN.Thuout[conthu] - HSI.Tcfinal0[j])) < 0.0001) {
								HSI.LMTDhu[j] = (CaseStudyHEN.Thuin[conthu] - CaseStudyHEN.Tcfinal[j]);
							}
							else {
								HSI.LMTDhu[j] = ((CaseStudyHEN.Thuin[conthu] - CaseStudyHEN.Tcfinal[j]) - (CaseStudyHEN.Thuout[conthu] - HSI.Tcfinal0[j])) / log(((CaseStudyHEN.Thuin[conthu] - CaseStudyHEN.Tcfinal[j]) / (CaseStudyHEN.Thuout[conthu] - HSI.Tcfinal0[j])));
							}
							HSI.Areahu[j] = HSI.Qhu[j] / (CaseStudyHEN.Uhu[j][conthu] * HSI.LMTDhu[j]);
							if (HSI.Areahu[j] > 0 && HSI.Areahu[j] < CaseStudyHEN.minarea) {
								HSI.TPenCosts = HSI.TPenCosts + 100 * CaseStudyHEN.taCtroc + CaseStudyHEN.taCtroc * pow((HSI.Areahu[j] - CaseStudyHEN.minarea), 2);
							}
							if (HSI.Areahu[j] < 0.0) {
								HSI.TPenCosts = HSI.TPenCosts + 100 * CaseStudyHEN.taCtroc + CaseStudyHEN.taCtroc * pow((HSI.Areahu[j]), 2);
							}
							else {
								HSI.AreaCosts = HSI.AreaCosts + (CaseStudyHEN.BB[(CaseStudyHEN.nhot - 1) + conthu][j] + CaseStudyHEN.CC[(CaseStudyHEN.nhot - 1) + conthu][j] * pow(HSI.Areahu[j], CaseStudyHEN.betaa[(CaseStudyHEN.nhot - 1) + conthu][j]));
							}
						}
					}
				}
			}
		}
	}
	// Utility Costs
	//UtilCosts = UtilCosts + HSI.HUcosts[i] * HSI.TotalQhu[i] + CUcosts[0] * HSI.TotalQcu;
	//PenaltyCosts...
	for (j = ncoldind; j >= 0; j--) {
		HSI.Tck[j][nstagesind] = CaseStudyHEN.Tcin[j];
		for (k = nstagesind; k >= 0; k--) {
			for (i = nhotind; i >= 0; i--) {
				if (HENSolution.z[i][j][k] == 1) {
					if (HENSolution.Q[i][j][k] < 0) {
						HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.qaCtroc + CaseStudyHEN.qbCtroc * pow((HENSolution.Q[i][j][k]), 2);
					}
					if (HENSolution.Fh[i][j][k] < 0.0 - CaseStudyHEN.Frmin || HENSolution.Fh[i][j][k] > 1.0 + CaseStudyHEN.Frmin) {
						HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.qaCtroc * 10 + pow((HENSolution.Fh[i][j][k]), 2);
					}
					if (HENSolution.Fc[i][j][k] < 0.0 - CaseStudyHEN.Frmin || HENSolution.Fc[i][j][k] > 1.0 + CaseStudyHEN.Frmin) {
						HSI.TPenCosts = HSI.TPenCosts + CaseStudyHEN.qaCtroc * 10 + pow((HENSolution.Fc[i][j][k]), 2);
					}
				}
			}
		}
	}

	double HUtotal = 0;
	for (j = 0; j <= ncoldind; j++) {
		for (int conthu = 0; conthu <= CaseStudyHEN.nhu - 1; conthu++) {
			if (HSI.Qhu[j] > Qmin) {
				HUtotal = HUtotal + HSI.Qhu[j];
			}
		}
	}
	for (i = 0; i <= nhotind; i++) {
		if (CaseStudyHEN.ishotutil[i] > 0.0) {
			for (k = 0; k <= nstages - 1; k++) {
				for (j = 0; j <= ncoldind; j++) {
					if (HENSolution.Q[i][j][k] > Qmin && HENSolution.Fh[i][j][k] > Frmin && HENSolution.Fc[i][j][k] > Frmin) {
						HUtotal = HUtotal + HENSolution.Q[i][j][k];
					}
				}
			}
		}
	}

	HENSolution.TotalCosts = CaseStudyHEN.AF * HSI.AreaCosts + UtilCosts + HSI.TPenCosts;
	HENSolution.TPenCosts = HSI.TPenCosts;

	if (isnan(HENSolution.TotalCosts)) {
		int aaaaaaaaa = 1;
	}

	HENSolution.HSI = HSI;

	if (print == 1) {
		time_t rawtime;
		struct tm timeinfo;
		char buffer[80];
		char buffer1[80];
		char buffer2[80];

		time(&rawtime);
		localtime_s(&timeinfo, &rawtime);


		sprintf_s(buffer, "%ih%ic%s %i-%i-%i %ih%im%is %i.txt", CaseStudyHEN.nhot, CaseStudyHEN.ncold, CaseStudyHEN.comment, timeinfo.tm_mday, timeinfo.tm_mon + 1, timeinfo.tm_year - 100, timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec, (int)HENSolution.TotalCosts);
		char QFF[500];
		ofstream myfile(buffer);
		if (myfile.is_open()) {
			sprintf_s(QFF, "Time = \t%.3f\n", 0.0);
			myfile << QFF;
			//sprintf_s(QFF, "cT0 = \t%.4f\tcL = \t%.4f\tT0 = \t%.4f\tL = \t%.4f\talpha =\t%.4f\tcalpha =\t%.4f\tPSOMaxIter =\t%.4f\n\n", RFOParam.cT0, cL, T0, tests[attcont][3], tests[attcont][6], tests[attcont][7], tests[attcont][8]);
			myfile << QFF;
			sprintf_s(QFF, "i\tj\tk\tQ\tFh\tFc\tThin\tThout\tTcin\tTcout\tLMTD\tArea\n");
			myfile << QFF;

			for (k = 0; k <= nstagesind; k++) {
				for (j = 0; j <= ncoldind; j++) {
					for (i = 0; i <= nhotind; i++) {
						if (HENSolution.Q[i][j][k] > Qmin) {
							sprintf_s(QFF, "%i\t%i\t%i\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t\n", i, j, k, HENSolution.Q[i][j][k], HENSolution.Fh[i][j][k], HENSolution.Fc[i][j][k], HENSolution.HSI.Thk[i][k], HENSolution.HSI.Thout[i][j][k], HENSolution.HSI.Tck[j][k], HENSolution.HSI.Tcout[i][j][k], HENSolution.HSI.LMTD[i][j][k], HENSolution.HSI.Area[i][j][k]);
							if (HENSolution.Fh[i][j][k] < 1.0 || HENSolution.Fc[i][j][k] < 1.0) {
								int aaaaaaa = 1;
							}
							//printf("Thin = %.4f | Tcin = %.4f \n", Thk[i][k], Tck[i][k]);
							//printf("Thout = %.4f | Tcout = %.4f \n", Thout[i][j][k], Tcout[i][j][k]);
							//printf("DTML = %.4f | Area = %.4f\n", LMTD[i][j][k], Area[i][j][k]);
							myfile << QFF;
						}
					}
				}
			}

			sprintf_s(QFF, "\ni\tQcu\tThin\tThout\tLMTDcu\tAreacu\n");
			myfile << QFF;
			for (i = 0; i <= nhotind; i++) {
				sprintf_s(QFF, "%i\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t\n", i, HENSolution.HSI.Qcu[i], HENSolution.HSI.Thfinal0[i], CaseStudyHEN.Thfinal[i], HENSolution.HSI.LMTDcu[i], HENSolution.HSI.Areacu[i]);
				myfile << QFF;
			}

			sprintf_s(QFF, "\nj\tQhu\tTcin\tTcout\tLMTDhu\tAreahu\n");
			myfile << QFF;
			for (j = 0; j <= ncoldind; j++) {
				sprintf_s(QFF, "%i\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t\n", j, HENSolution.HSI.Qhu[j], HENSolution.HSI.Tcfinal0[j], CaseStudyHEN.Tcfinal[j], HENSolution.HSI.LMTDhu[j], HENSolution.HSI.Areahu[j]);
				myfile << QFF;
			}

			sprintf_s(QFF, "\nTotal Costs = \t%.12f", HENSolution.TotalCosts);
			myfile << QFF;

			sprintf_s(QFF, "\nInput:\n\n");
			myfile << QFF;
			for (k = 0; k <= nstagesind; k++) {
				for (j = 0; j <= ncoldind; j++) {
					for (i = 0; i <= nhotind; i++) {
						if (HENSolution.Q[i][j][k] > Qmin) {
							sprintf_s(QFF, "i=%i; j=%i; k=%i; Particle[p].Q[i][j][k]=%.12f; Particle[p].Fh[i][j][k]=%.12f; Particle[p].Fc[i][j][k]=%.12f;\n", i, j, k, HENSolution.Q[i][j][k], HENSolution.Fh[i][j][k], HENSolution.Fc[i][j][k]);
							//printf("Thin = %.4f | Tcin = %.4f \n", Thk[i][k], Tck[i][k]);
							//printf("Thout = %.4f | Tcout = %.4f \n", Thout[i][j][k], Tcout[i][j][k]);
							//printf("DTML = %.4f | Area = %.4f\n", LMTD[i][j][k], Area[i][j][k]);
							myfile << QFF;
						}
					}
				}
			}

			sprintf_s(QFF, "\nFor use with WHEN\n");
			myfile << QFF;
			for (i = 0; i <= nhotind; i++) {
				sprintf_s(QFF, "Particle[p].Qcuinter[%i][0][nstagesind] = %.15f; Particle[p].Fhcuinter[%i][0][nstagesind] = 1.0;\n", i, HENSolution.HSI.Qcu[i], i);
				myfile << QFF;
			}
			for (j = 0; j <= ncoldind; j++) {
				sprintf_s(QFF, "Particle[p].Qhuinter[0][%i][0] = %.15f; Particle[p].Fchuinter[0][%i][0] = 1.0;\n", j, HENSolution.HSI.Qhu[j], j);
				myfile << QFF;
			}

			myfile.close();
		}
		else cout << "Unable to open file";

		//double end = 0.0;// clock();
		//time = 0.0;// (double)(end - start) / CLOCKS_PER_SEC;

		//cout << time << endl;
		myfile.close();
	}

}

struct SAParamStruct {
	double T0;
	double Tf;
	int L;
	double alpha;

	int fixtopology;
	int fixtopologyplus;

	vector<vector<vector<int>>> fixedz;
	vector<int> Forbidh;
	vector<int> Forbidc;

	int maxHE;
};

struct RFOParamStruct {
	double cT0;
	double cTf;
	int cL;
	double slowingfactor;
	double calpha;

	int Particles;
	int PSOMaxIter;
	double c1;
	double c2;
	double wmin;
	double wmax;
	double v0factor;
	double v0ffactor;
};

void SARFO(int inputsol, int spaghetti, int nstages, SAParamStruct SAParam, RFOParamStruct RFOParam, CaseStudyHENStruct CaseStudyHEN, HENSynInterm& HSI, HENSolutionStruct& HENSolution, vector<int> today, double start) {
	int endopt = 0;
	HENSynInterm HSIzero;
	HSIzero = HSI;

	double Tk = SAParam.T0;
	double Tf = SAParam.Tf;
	int l = 0;
	int L = SAParam.L;
	int fixtopology = SAParam.fixtopology;
	int fixtopologyplus = 0;
	int attcont = 0;
	int nstagesind = nstages - 1;
	int nhotind = CaseStudyHEN.nhot + CaseStudyHEN.nhu - 1;
	int ncoldind = CaseStudyHEN.ncold + CaseStudyHEN.ncu - 1;
	int nhot = nhotind + 1;
	int ncold = ncoldind + 1;
	int i = 0; int j = 0; int k = 0;
	int x = 0;
	int p = 0;
	int Particles = RFOParam.Particles;
	int tourlength = 0;
	int nhe = 0;
	int maxHE = SAParam.maxHE;
	double Qmin = CaseStudyHEN.Qmin;
	double Frmin = CaseStudyHEN.Frmin;
	int tourlengthind = 0;
	int tourindbestlength = 0;
	int tourindsollength = 0;
	int firstsol = 1;
	int hasfoundnonpen = 0;
	int bestnhe = 0;
	int nonacceptcont = 0;
	double bestsoltime = 0;

	HENSolutionStruct bestsol;
	HENSolutionStruct sol;
	HENSolutionStruct NewSol;
	HENSolutionStruct solContVars;
	HENSolutionStruct bestsolCont;
	vector<HENSolutionStruct> Particle;
	Particle.resize(Particles);
	vector<HENSolutionStruct> ParticleBest;
	ParticleBest.resize(Particles);
	HENSolutionStruct GlobalBest;

	for (p = 0; p <= Particles - 1; p++) {
		Particle[p] = HENSolution;
	}
	bestsol = HENSolution;
	sol = HENSolution;
	NewSol = HENSolution;
	solContVars = HENSolution;
	bestsolCont = HENSolution;
	HENSolutionStruct InitialSol = HENSolution;

	time_t rawtime;
	struct tm timeinfo;
	char buffer[80];
	char buffer1[80];
	char buffer2[80];

	time(&rawtime);
	localtime_s(&timeinfo, &rawtime);

	sprintf_s(buffer2, "Solutions\\%ih%ic %i-%i-%i %ih%im%is TACxTIMExNHE.txt", nhot, ncold, timeinfo.tm_mday, timeinfo.tm_mon + 1, timeinfo.tm_year - 100, timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);

	//ofstream myfile1(buffer1);
	ofstream myfile2(buffer2);

	vector<vector<int>> tourindbest;
	tourindbest.resize(3);
	while (Tk >= Tf) {
		for (l = 0; l < L; l++) {
			p = 0;

			vector<vector<int>> tourind;
			tourind.resize(3);
			vector<vector<int>> tourindsol;
			tourind.resize(3);

			int randomh = 0;
			int randomc = 0;
			int randoms = 0;
			if ((fixtopology == 1 && attcont > -1)) {
				if (fixtopologyplus == 0) {
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								NewSol.z[i][j][k] = SAParam.fixedz[i][j][k];
							}
						}
					}
					x = 0;
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								if (NewSol.z[i][j][k] == 1) {
									tourind[0].push_back(i);
									tourind[1].push_back(j);
									tourind[2].push_back(k);
									x++;
								}
							}
						}
					}
					int rrrr = 0;
					rrrr = rand() % x;
					randomh = tourind[0][rrrr];
					randomc = tourind[1][rrrr];
					randoms = tourind[2][rrrr];
				}
				else {
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								if (SAParam.fixedz[i][j][k] == 1) {
									NewSol.z[i][j][k] = SAParam.fixedz[i][j][k];
								}
							}
						}
					}
					randomh = rand() % nhot;
					while (SAParam.Forbidh[randomh] == 1) {
						randomh = rand() % nhot;
					}
					randomc = rand() % ncold;
					while (SAParam.Forbidc[randomc] == 1) {
						randomc = rand() % ncold;
					}
					randoms = rand() % nstages;

					NewSol.z[randomh][randomc][randoms] = 1;
					tourlength++;
					nhe = tourlength;
				}
			}
			else {
				randomh = rand() % nhot;
				while (SAParam.Forbidh[randomh] == 1) {
					randomh = rand() % nhot;
				}
				randomc = rand() % ncold;
				while (SAParam.Forbidc[randomc] == 1) {
					randomc = rand() % ncold;
				}
				randoms = rand() % nstages;

				NewSol.z[randomh][randomc][randoms] = 1;
				tourlength++;
				nhe = tourlength;

				if (nhe > maxHE) {
					x = 0;
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								if (NewSol.z[i][j][k] == 1) {
									tourind[0].push_back(i);
									tourind[1].push_back(j);
									tourind[2].push_back(k);
									x++;
								}
							}
						}
					}
					int rrrr = 0;
					rrrr = rand() % x;
					randomh = tourind[0][rrrr];
					randomc = tourind[1][rrrr];
					randoms = tourind[2][rrrr];

					NewSol.z[randomh][randomc][randoms] = 0;
					tourlength--;
					nhe = tourlength;
				}


			}

			vector<vector<vector<int>>> varybin;
			varybin.resize(CaseStudyHEN.AllHotStreams.size());
			for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
				varybin[i].resize(CaseStudyHEN.AllColdStreams.size());
				for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
					varybin[i][j].resize(nstages);
				}
			}
			//identify new matchs group
			x = 0;
			for (k = 0; k <= nstagesind; k++) {
				for (j = 0; j <= ncoldind; j++) {
					for (i = 0; i <= nhotind; i++) {
						varybin[i][j][k] = 0;
						if (NewSol.z[i][j][k] == 1) {
							tourind[0].push_back(i);
							tourind[1].push_back(j);
							tourind[2].push_back(k);
							x++;
						}
					}
				}
			}

			nhe = x;
			tourlength = x;
			tourlengthind = x - 1;

			tourindbest.clear();
			tourindbest.resize(3);
			int xcont;
			xcont = 0;
			for (k = 0; k <= nstagesind; k++) {
				for (j = 0; j <= ncoldind; j++) {
					for (i = 0; i <= nhotind; i++) {
						if (bestsol.Q[i][j][k] > Qmin && bestsol.Fh[i][j][k] > Frmin && bestsol.Fc[i][j][k] > Frmin) {
							tourindbest[0].push_back(i);
							tourindbest[1].push_back(j);
							tourindbest[2].push_back(k);
							xcont++;
						}
					}
				}
			}
			tourindbestlength = xcont - 1;

			tourindsol.clear();
			tourindsol.resize(3);
			xcont = 0;
			for (k = 0; k <= nstagesind; k++) {
				for (j = 0; j <= ncoldind; j++) {
					for (i = 0; i <= nhotind; i++) {
						if (sol.Q[i][j][k] > Qmin && sol.Fh[i][j][k] > Frmin && sol.Fc[i][j][k] > Frmin) {
							tourindsol[0].push_back(i);
							tourindsol[1].push_back(j);
							tourindsol[2].push_back(k);
							xcont++;
						}
					}
				}
			}
			tourindsollength = xcont - 1;


			vector<vector<int>> varybinaux1;
			varybinaux1.resize(tourlengthind + 1);
			for (int co1 = 0; co1 <= tourlengthind; co1++) {
				varybinaux1[co1].resize(tourlengthind + 1);
				for (int co2 = 0; co2 <= tourlengthind; co2++) {
					varybinaux1[co1][co2] = 0;
				}
			}
			for (int co1 = 0; co1 <= tourlengthind; co1++) {
				for (int co2 = 0; co2 <= tourlengthind; co2++) {
					if (tourind[0][co1] == tourind[0][co2] || tourind[1][co1] == tourind[1][co2]) {
						varybinaux1[co1][co2] = 1;
					}
					else {
						varybinaux1[co1][co2] = 0;
					}
				}
			}

			vector<vector<int>> varybinaux2;
			varybinaux2.resize(tourlengthind + 1);
			for (int co1 = 0; co1 <= tourlengthind; co1++) {
				varybinaux2[co1].resize(tourlengthind + 1);
				for (int co2 = 0; co2 <= tourlengthind; co2++) {
					varybinaux2[co1][co2] = 0;
				}
			}
			for (int co1 = 0; co1 <= tourlengthind; co1++) {
				for (int co2 = 0; co2 <= tourlengthind; co2++) {
					for (int co3 = 0; co3 <= tourlengthind; co3++) {
						if ((varybinaux1[co3][co2] == 1) && (varybinaux1[co1][co2] == 1)) {
							for (int co4 = 0; co4 <= tourlengthind; co4++) {
								varybinaux2[co1][co4] = varybinaux2[co1][co4] + varybinaux1[co3][co4];
							}
						}
					}
				}
			}

			//for (co1 = 0; co1 <= tourlengthind; co1++) {
				//for (co2 = 0; co2 <= tourlengthind; co2++) {
				//	cout << varybinaux2[co1][co2] << " ";
				//}
				//cout << endl;
			//}
			//cout << endl;
			vector<vector<int>> varybinaux3;
			varybinaux3.resize(tourlengthind + 1);
			for (int co1 = 0; co1 <= tourlengthind; co1++) {
				varybinaux3[co1].resize(tourlengthind + 1);
				for (int co2 = 0; co2 <= tourlengthind; co2++) {
					varybinaux3[co1][co2] = 0;
				}
			}
			for (int co1 = 0; co1 <= tourlengthind; co1++) {
				for (int co2 = 0; co2 <= tourlengthind; co2++) {
					for (int co3 = 0; co3 <= tourlengthind; co3++) {
						if ((varybinaux2[co3][co2] > 0) && (varybinaux2[co1][co2] > 0)) {
							for (int co4 = 0; co4 <= tourlengthind; co4++) {
								varybinaux3[co1][co4] = varybinaux3[co1][co4] + varybinaux2[co3][co4];
							}
						}
					}
				}
			}
			for (int co1 = 0; co1 <= tourlengthind; co1++) {
				for (int co2 = 0; co2 <= tourlengthind; co2++) {
					if (varybinaux3[co1][co2] > 0) {
						varybinaux3[co1][co2] = 1;
					}
					//		cout << varybinaux3[co1][co2] << " ";
				}
				//cout << endl;
			}
			//	cout << endl;
			int varyind = 0;
			for (int co1 = 0; co1 <= tourlengthind; co1++) {
				if (tourind[0][co1] == randomh && tourind[1][co1] == randomc && tourind[2][co1] == randoms) {
					varyind = co1;
				}
			}
			vector<vector<int>> varyline;
			varyline.resize(1);
			for (int co1 = 0; co1 <= tourlengthind; co1++) {
				varyline[0].push_back(0);
				//varyline[0][co1] = 0;
			}
			int whco = 0;
			int co1 = 0;
			while (whco == 0) {
				if (varybinaux3[co1][varyind] == 1) {
					for (int co2 = 0; co2 <= tourlengthind; co2++) {
						varyline[0][co2] = varybinaux3[co1][co2];
					}
					whco = 1;
				}
				co1++;
			}
			int tourlengthvarind = 0;
			int co2 = 0;
			vector<vector<int>> tourindvar; tourindvar.resize(3);
			for (co1 = 0; co1 <= tourlengthind; co1++) {
				if (varyline[0][co1] == 1) {
					varybin[tourind[0][co1]][tourind[1][co1]][tourind[2][co1]] = 1;
					tourindvar[0].push_back(tourind[0][co1]);
					tourindvar[1].push_back(tourind[1][co1]);
					tourindvar[2].push_back(tourind[2][co1]);
					tourlengthvarind++;
					co2++;
				}
			}
			int tourlengthvar = tourlengthvarind;
			tourlengthvarind = tourlengthvar - 1;


			printf("\n\nBest sol. topology:");


			cout << endl;
			for (co1 = 0; co1 <= 2; co1++) {
				for (co2 = 0; co2 <= tourindbestlength; co2++) {
					cout << tourindbest[co1][co2] + 1 << " ";
				}
				cout << endl;
			}
			cout << endl;

			printf("Current sol. topology:");


			cout << endl;
			for (co1 = 0; co1 <= 2; co1++) {
				for (co2 = 0; co2 <= tourindsollength; co2++) {
					cout << tourindsol[co1][co2] + 1 << " ";
				}
				cout << endl;
			}
			cout << endl;

			printf("New Match = %i %i %i\n", randomh + 1, randomc + 1, randoms + 1);
			printf("New topology:");

			cout << endl;
			for (co1 = 0; co1 <= 2; co1++) {
				for (co2 = 0; co2 <= tourlengthind; co2++) {
					cout << tourind[co1][co2] + 1 << " ";
				}
				cout << endl;
			}
			for (co1 = 0; co1 <= tourlengthind; co1++) {
				cout << varyline[0][co1] << " ";

			}
			cout << endl;

			for (k = 0; k <= nstagesind; k++) {
				for (i = 0; i <= nhotind; i++) {
					for (j = 0; j <= ncoldind; j++) {
						if (NewSol.z[i][j][k] == 1) {
							if (varybin[i][j][k] == 1) {
								Particle[p].Q[i][j][k] = 0;
							}
							else {
								Particle[p].Q[i][j][k] = sol.Q[i][j][k];
							}
						}
					}
				}
			}
			//Initializing Fractions
			//Hot Stream Splits
			for (k = 0; k <= nstagesind; k++) {
				for (i = 0; i <= nhotind; i++) {
					double nhsplitsaux = 0;
					for (j = 0; j <= ncoldind; j++) {
						if (varybin[i][j][k] == 1) {
							nhsplitsaux = nhsplitsaux + 1;
						}
					}
					double nhsplitsaux2 = 1 / nhsplitsaux;
					for (j = 0; j <= ncoldind; j++) {
						if (NewSol.z[i][j][k] == 1) {
							if (varybin[i][j][k] == 1) {
								Particle[p].Fh[i][j][k] = nhsplitsaux2;
							}
							else {
								Particle[p].Fh[i][j][k] = sol.Fh[i][j][k];
							}
						}
					}
				}
			}

			//Cold Stream Splits
			for (k = 0; k <= nstagesind; k++) {
				for (j = 0; j <= ncoldind; j++) {
					double ncsplitsaux = 0;
					for (i = 0; i <= nhotind; i++) {
						if (varybin[i][j][k] == 1) {
							ncsplitsaux = ncsplitsaux + 1;
						}
					}
					double ncsplitsaux2 = 1 / ncsplitsaux;
					for (i = 0; i <= nhotind; i++) {
						if (NewSol.z[i][j][k] == 1) {
							if (varybin[i][j][k] == 1) {
								Particle[0].Fc[i][j][k] = ncsplitsaux2;
							}
							else {
								Particle[0].Fc[i][j][k] = sol.Fc[i][j][k];
							}
						}
					}
				}
			}

			//Number of Hot Stream Splits ARRUMAR AQUI O NHSPLITS, N�O DEVERIA RESETAR
			x = 0;
			double nhsplits = 0;
			vector<vector<int>> Fhtourind; Fhtourind.resize(3);
			for (k = 0; k <= nstagesind; k++) {

				for (i = 0; i <= nhotind; i++) {
					double nhsplitsaux = 0;
					for (j = 0; j <= ncoldind; j++) {
						if (varybin[i][j][k] == 1) {
							nhsplitsaux = nhsplitsaux + 1;
						}
					}
					if (nhsplitsaux > 1) {
						nhsplits = nhsplits + nhsplitsaux;
						for (int jj = 0; jj <= ncoldind; jj++) {
							if (varybin[i][jj][k] == 1) {
								Fhtourind[0].push_back(i);
								Fhtourind[1].push_back(jj);
								Fhtourind[2].push_back(k);
								x++;
							}
						}
					}
				}
			}
			//Number of Cold Stream Splits

			x = 0;
			double ncsplits = 0;
			vector<vector<int>> Fctourind; Fctourind.resize(3);
			for (k = 0; k <= nstagesind; k++) {

				for (j = 0; j <= ncoldind; j++) {
					double ncsplitsaux = 0;
					for (i = 0; i <= nhotind; i++) {
						if (varybin[i][j][k] == 1) {
							ncsplitsaux = ncsplitsaux + 1;
						}
					}
					if (ncsplitsaux > 1) {
						ncsplits = ncsplits + ncsplitsaux;
						for (int ii = 0; ii <= nhotind; ii++) {
							if (varybin[ii][j][k] == 1) {
								Fctourind[0].push_back(ii);
								Fctourind[1].push_back(j);
								Fctourind[2].push_back(k);
								x++;
							}
						}
					}

				}
			}

			//Number of HE (This will probably be counted in outter SA. Using 9 for tests)
			int totalvars = nhsplits + ncsplits + nhe;
			vector<double> VRoul; VRoul.resize(3);
			if (totalvars == 0 || nhe == 0) {
				VRoul[0] = 1;
				VRoul[1] = 0;
				VRoul[2] = 0;
			}
			else {
				VRoul[0] = (double)nhe / (double)totalvars;
				VRoul[1] = (double)nhsplits / (double)totalvars;
				VRoul[2] = (double)ncsplits / (double)totalvars;
			}


			double sumVarRouletteAux;
			double sumVarRoulette[3] = {};
			sumVarRouletteAux = VRoul[0];
			sumVarRoulette[0] = VRoul[0];

			for (x = 1; x <= 2; x++) {
				sumVarRoulette[x] = VRoul[x] + sumVarRoulette[x - 1];
				//sumVarRouletteAux = sumVarRouletteAux + VRoul[x];
				//sumVarRoulette[x] = sumVarRouletteAux;
			}

			vector<vector<vector<double>>> Qmaxm1;
			vector<vector<vector<double>>> Qmax0;
			vector<vector<vector<double>>> Qmax;
			Qmaxm1.resize(CaseStudyHEN.AllHotStreams.size());
			Qmax0.resize(CaseStudyHEN.AllHotStreams.size());
			Qmax.resize(CaseStudyHEN.AllHotStreams.size());
			for (int i = 0; i <= CaseStudyHEN.AllHotStreams.size() - 1; i++) {
				Qmaxm1[i].resize(CaseStudyHEN.AllColdStreams.size());
				Qmax0[i].resize(CaseStudyHEN.AllColdStreams.size());
				Qmax[i].resize(CaseStudyHEN.AllColdStreams.size());
				for (int j = 0; j <= CaseStudyHEN.AllColdStreams.size() - 1; j++) {
					Qmaxm1[i][j].resize(nstages);
					Qmax0[i][j].resize(nstages);
					Qmax[i][j].resize(nstages);
				};
			};
			//Calculate Qmax0
			for (k = 0; k <= nstagesind; k++) {
				for (j = 0; j <= ncoldind; j++) {
					for (i = 0; i <= nhotind; i++) {
						if (NewSol.z[i][j][k] == 1) {
							if (CaseStudyHEN.Qh[i] > CaseStudyHEN.Qc[j]) {
								if (CaseStudyHEN.iscoldutil[j] > 0.0) {
									Qmax0[i][j][k] = CaseStudyHEN.Qh[i];
									Qmax[i][j][k] = Qmax0[i][j][k];
								}
								else {
									Qmax0[i][j][k] = CaseStudyHEN.Qc[j];
									Qmax[i][j][k] = Qmax0[i][j][k]; //QMAX0 � O PROBLEMA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
								}
							}
							else {
								if (CaseStudyHEN.ishotutil[i] > 0.0) {
									Qmax0[i][j][k] = CaseStudyHEN.Qc[j];
									Qmax[i][j][k] = Qmax0[i][j][k]; //QMAX0 � O PROBLEMA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
								}
								else {
									Qmax0[i][j][k] = CaseStudyHEN.Qh[i];
									Qmax[i][j][k] = Qmax0[i][j][k];
								}
							}
						}
					}
				}
			}


			/*
			x = 1;
			double rl = 0;
			for (x = 1; x <= 10; x++) {
			RI ri(0, 3);
			rl = ri(g);
			}
			*/

			/*
			RI ri(0, tourlengthind);
			FCRI fcri(1, nhsplits-1);
			FHRI fhri(1, ncsplits-1);
			fixFHRI fixfhri(0, 0);
			fixFCRI fixfcri(0, 0);
			*/

			//===============================================================================
			//=== SIMULATED ANNEALING =======================================================
			//===============================================================================

			double ball = (double)rand() / (double)RAND_MAX;//d(g);			
			int bestsolreuseflag = 0;
			int reuseflag = 0;
			if (ball <= 0.2 || bestsolreuseflag == 1) {
				reuseflag = 1;
			}
			else {
				reuseflag = 0;
			}

			//if (ActivateSpaghetti == 1 && finalattempt == 1) {
			//	reuseflag = 1;
			//}
			//if (ActivateSpaghetti == 1 && numreuses <= 10) {
			//	reuseflag = 1;
			//	numreuses++;
			//}

			double cT = RFOParam.cT0;
			int firstcalc = 1;
			Particle[0].z = NewSol.z;
			SWS(0, nstages, CaseStudyHEN, Particle[0], HSI, 0);
			/*
			for (int i = 0; i <= HENSolution.Q.size() - 1; i++) {
				for (int j = 0; j <= HENSolution.Q[i].size() - 1; j++) {
					for (int k = 0; k <= HENSolution.Q[i][j].size() - 1; k++) {
						Particle[0].Fc[i][j][k] = 0;
						Particle[0].Fh[i][j][k] = 0;
						Particle[0].Q[i][j][k] = 0;
					}
				}
			}
			*/
			if (spaghetti == 0) {
				int aaaaaa = 0;
			}
			double Fhsum = 0;
			double Fcsum = 0;
			while (cT > RFOParam.cTf) {
				for (int cLcont = 0; cLcont <= RFOParam.cL; cLcont++) {
					int found = 0;
					int MoveVar = 0;
					int moveindex = 0;
					double contmove = 0;

					ball = (double)rand() / (double)RAND_MAX;//d(g);
					x = 0;
					if (ball == 1) {
						ball = 0.9999999999;
					}
					while (found == 0) {
						if (ball < sumVarRoulette[x]) { //VARI�VEL TODA ERRADA!!!!
							MoveVar = x;
							found = 1;
						}
						x++;
					}
					//MoveVar = 0;
					/*
					cout << "0 for Q; 1 for Fh; 2 for Fc ";
					cin >> MoveVar;
					*/
					int QmaxnotOKsum = 0;
					vector<vector<int>> QmaxnotOK; QmaxnotOK.resize(3);
					vector<vector<int>> Fhtourind2; Fhtourind2.resize(3);
					vector<vector<int>> Fctourind2; Fctourind2.resize(3);
					double Fhmove = 0;
					int Fhmoveind = 0;
					double Fcmove = 0;
					int Fcmoveind = 0;
					int jj = 0;
					int remfracfrom = 0;
					int ii = 0;
					double Qhkaux;
					double Qckaux;
					switch (MoveVar) {
					case 0:
						moveindex = rand() % tourlengthvar; //ri(g);
						contmove = (((double)rand() / (double)RAND_MAX) - 0.5) * 2 * Qmax0[tourindvar[0][moveindex]][tourindvar[1][moveindex]][tourindvar[2][moveindex]] * exp((cT - RFOParam.cT0) / (RFOParam.cT0 * RFOParam.slowingfactor));
						/*
						cout << "tourind ";
						cin >> moveindex;
						cout << "Q ";
						cin >> contmove;
						*/
						Particle[0].Q[tourindvar[0][moveindex]][tourindvar[1][moveindex]][tourindvar[2][moveindex]] = Particle[0].Q[tourindvar[0][moveindex]][tourindvar[1][moveindex]][tourindvar[2][moveindex]] + contmove;

						/*
						if (Particle[0].Q[tourind[0][moveindex]][tourind[1][moveindex]][tourind[2][moveindex]] < 0) {
						TPenCosts = TPenCosts + qaCtroc + qbCtroc*pow((Particle[0].Q[tourind[0][moveindex]][tourind[1][moveindex]][tourind[2][moveindex]]), 2);
						}
						*/

						//Qhk = Qh - sum(sum(Particle(p).Q, 3), 2);
						Qhkaux = 0;
						for (i = 0; i <= nhotind; i++) {
							for (k = 0; k <= nstagesind; k++) {
								for (j = 0; j <= ncoldind; j++) {
									if (Particle[p].Q[i][j][k] > Qmin || Particle[p].Q[i][j][k] < -Qmin) {
										Qhkaux = Qhkaux + Particle[p].Q[i][j][k];
									};
								};
							};
							CaseStudyHEN.Qhk[i] = CaseStudyHEN.Qh[i] - Qhkaux;
							Qhkaux = 0;
						};


						//Qck = Qc - sum(sum(Particle(p).Q, 3), 1)';
						Qckaux = 0;
						for (j = 0; j <= ncoldind; j++) {
							for (k = 0; k <= nstagesind; k++) {
								for (i = 0; i <= nhotind; i++) {
									if (Particle[p].Q[i][j][k] > Qmin || Particle[p].Q[i][j][k] < -Qmin) {
										Qckaux = Qckaux + Particle[p].Q[i][j][k];
									};
								};
							};
							CaseStudyHEN.Qck[j] = CaseStudyHEN.Qc[j] - Qckaux;
							Qckaux = 0;
						};

						for (k = 0; k <= nstagesind; k++) {
							for (j = 0; j <= ncoldind; j++) {
								for (i = 0; i <= nhotind; i++) {
									if (NewSol.z[i][j][k] == 1) {
										if (CaseStudyHEN.Qhk[i] > CaseStudyHEN.Qck[j]) {
											Qmax[i][j][k] = CaseStudyHEN.Qck[j];
										}
										else {
											Qmax[i][j][k] = CaseStudyHEN.Qhk[i];
										}
									}
								}
							}
						}

						//Check if some Qmax is < 0
						QmaxnotOKsum = 0;

						for (k = 0; k <= nstagesind; k++) {
							for (j = 0; j <= ncoldind; j++) {
								for (i = 0; i <= nhotind; i++) {
									if (Qmax[i][j][k] < 0 && Particle[p].Q[i][j][k] > 0) {
										QmaxnotOK[0].push_back(i);
										QmaxnotOK[1].push_back(j);
										QmaxnotOK[2].push_back(k);
										QmaxnotOKsum = QmaxnotOKsum + 1;
									}
								}
							}
						}
						while (QmaxnotOKsum > 0) {
							//fixRI fixri(0, QmaxnotOKsum);
							int indtofix = rand() % QmaxnotOKsum;//fixri(g);

							Particle[p].Q[QmaxnotOK[0][indtofix]][QmaxnotOK[1][indtofix]][QmaxnotOK[2][indtofix]] = Particle[p].Q[QmaxnotOK[0][indtofix]][QmaxnotOK[1][indtofix]][QmaxnotOK[2][indtofix]] + Qmax[QmaxnotOK[0][indtofix]][QmaxnotOK[1][indtofix]][QmaxnotOK[2][indtofix]];

							//Qhk = Qh - sum(sum(Particle(p).Q, 3), 2);
							Qhkaux = 0;
							for (i = 0; i <= nhotind; i++) {
								for (k = 0; k <= nstagesind; k++) {
									for (j = 0; j <= ncoldind; j++) {
										if (Particle[p].Q[i][j][k] > Qmin) {
											Qhkaux = Qhkaux + Particle[p].Q[i][j][k];
										};
									};
								};
								CaseStudyHEN.Qhk[i] = CaseStudyHEN.Qh[i] - Qhkaux;
								Qhkaux = 0;
							};


							//Qck = Qc - sum(sum(Particle(p).Q, 3), 1)';
							for (j = 0; j <= ncoldind; j++) {
								for (k = 0; k <= nstagesind; k++) {
									for (i = 0; i <= nhotind; i++) {
										if (Particle[p].Q[i][j][k] > Qmin) {
											Qckaux = Qckaux + Particle[p].Q[i][j][k];
										};
									};
								};
								CaseStudyHEN.Qck[j] = CaseStudyHEN.Qc[j] - Qckaux;
								Qckaux = 0;
							};

							//Update Qmax
							for (k = 0; k <= nstagesind; k++) {
								for (j = 0; j <= ncoldind; j++) {
									for (i = 0; i <= nhotind; i++) {
										if (NewSol.z[i][j][k] == 1) {
											if (CaseStudyHEN.Qhk[i] > CaseStudyHEN.Qck[j]) {
												Qmax[i][j][k] = CaseStudyHEN.Qck[j];
											}
											else {
												Qmax[i][j][k] = CaseStudyHEN.Qhk[i];
											}
										}
									}
								}
							}

							//Se a soma de (Qmax < 0) & (particle.Q > Qmin) >= 1
							//Se a matriz (contmove > 0) for igual a (Qmax < 0 & Particle(p).Q > Qmin)
							//O unico possivel de se retirar calor � ele mesmo
							//else, sortear
							//indices possiveis sao os em que Qmax < 0, Q > Qmin e contmove = 0

							//

							//Check if some Qmax is < 0
							QmaxnotOK[0].clear(); QmaxnotOK[1].clear(); QmaxnotOK[2].clear();
							QmaxnotOKsum = 0;
							for (k = 0; k <= nstagesind; k++) {
								for (j = 0; j <= ncoldind; j++) {
									for (i = 0; i <= nhotind; i++) {
										if (Qmax[i][j][k] < 0 && Particle[p].Q[i][j][k] > Qmin) {
											QmaxnotOK[0].push_back(i);
											QmaxnotOK[1].push_back(j);
											QmaxnotOK[2].push_back(k);
											QmaxnotOKsum = QmaxnotOKsum + 1;
										}
									}
								}
							}
						}
						/*
						for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
						for (i = 0; i <= nhotind; i++) {
						if (z[i][j][k] == 1) {
						cout << " " << Qmax0[i][j][k];
						}
						}
						}
						}
						cout << endl;
						for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
						for (i = 0; i <= nhotind; i++) {
						if (z[i][j][k] == 1) {
						cout << " " << Particle[0].Q[i][j][k];
						}
						}
						}
						}
						cout << endl;
						for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
						for (i = 0; i <= nhotind; i++) {
						if (z[i][j][k] == 1) {
						cout << " " << Qmax[i][j][k];
						}
						}
						}
						}


						cout << endl;
						system("pause");
						*/
						break;
					case 1:
						Fhmove = 0.1 * (((double)rand() / (double)RAND_MAX) - 0.5) * 2 * exp((cT - RFOParam.cT0) / (RFOParam.cT0 * RFOParam.slowingfactor));
						Fhmoveind = rand() % (int)nhsplits; //fcri(g);

						Particle[p].Fh[Fhtourind[0][Fhmoveind]][Fhtourind[1][Fhmoveind]][Fhtourind[2][Fhmoveind]] = Particle[p].Fh[Fhtourind[0][Fhmoveind]][Fhtourind[1][Fhmoveind]][Fhtourind[2][Fhmoveind]] + Fhmove;

						for (int jj = 0; jj <= ncoldind; jj++) {
							if (Particle[p].Fh[Fhtourind[0][Fhmoveind]][jj][Fhtourind[2][Fhmoveind]] > (1 - Frmin)) {
								Particle[p].Fh[Fhtourind[0][Fhmoveind]][jj][Fhtourind[2][Fhmoveind]] = 1;
							}
							if (Particle[p].Fh[Fhtourind[0][Fhmoveind]][jj][Fhtourind[2][Fhmoveind]] < Frmin) {
								Particle[p].Fh[Fhtourind[0][Fhmoveind]][jj][Fhtourind[2][Fhmoveind]] = 0;
							}
						}

						Fhsum = 0;
						for (int jj = 0; jj <= ncoldind; jj++) {
							Fhsum = Fhsum + Particle[p].Fh[Fhtourind[0][Fhmoveind]][jj][Fhtourind[2][Fhmoveind]];
						}

						while (abs(1 - Fhsum) > Frmin) {
							x = 0;
							jj = 0;
							Fhtourind2[0].clear(); Fhtourind2[1].clear(); Fhtourind2[2].clear();
							for (x = 0; x <= nhsplits - 1; x++) {
								if (Fhtourind[0][x] == Fhtourind[0][Fhmoveind] && Fhtourind[2][x] == Fhtourind[2][Fhmoveind]) {
									Fhtourind2[0].push_back(Fhtourind[0][x]);
									Fhtourind2[1].push_back(Fhtourind[1][x]);
									Fhtourind2[2].push_back(Fhtourind[2][x]);
									jj++;
								}
							}
							//fixFHRI fixfhri(0, jj - 1);
							Fhmoveind = rand() % jj;// fixfhri(g);
							Particle[p].Fh[Fhtourind2[0][Fhmoveind]][Fhtourind2[1][Fhmoveind]][Fhtourind2[2][Fhmoveind]] = Particle[p].Fh[Fhtourind2[0][Fhmoveind]][Fhtourind2[1][Fhmoveind]][Fhtourind2[2][Fhmoveind]] - (Fhsum - 1);

							//	Fhsum = 0;
							//	for (jj = 0; jj <= ncoldind; jj++) {
							//		Fhsum = Fhsum + Particle[p].Fh[Fhtourind2[0][Fhmoveind]][jj][Fhtourind2[2][Fhmoveind]];
							//	}
							for (jj = 0; jj <= ncoldind; jj++) {
								if (Particle[p].Fh[Fhtourind2[0][Fhmoveind]][jj][Fhtourind2[2][Fhmoveind]] > (1 - Frmin)) {
									Particle[p].Fh[Fhtourind2[0][Fhmoveind]][jj][Fhtourind2[2][Fhmoveind]] = 1;
								}
								if (Particle[p].Fh[Fhtourind2[0][Fhmoveind]][jj][Fhtourind2[2][Fhmoveind]] < Frmin) {
									Particle[p].Fh[Fhtourind2[0][Fhmoveind]][jj][Fhtourind2[2][Fhmoveind]] = 0;
									Particle[p].Q[Fhtourind2[0][Fhmoveind]][jj][Fhtourind2[2][Fhmoveind]] = 0;
								}
							}
							Fhsum = 0;
							for (jj = 0; jj <= ncoldind; jj++) {
								Fhsum = Fhsum + Particle[p].Fh[Fhtourind2[0][Fhmoveind]][jj][Fhtourind2[2][Fhmoveind]];
							}
						}
						break;
					default:
						Fcmove = 0.1 * (((double)rand() / (double)RAND_MAX) - 0.5) * 2 * exp((cT - RFOParam.cT0) / (RFOParam.cT0 * RFOParam.slowingfactor));
						Fcmoveind = rand() % (int)ncsplits;

						Particle[p].Fc[Fctourind[0][Fcmoveind]][Fctourind[1][Fcmoveind]][Fctourind[2][Fcmoveind]] = Particle[p].Fc[Fctourind[0][Fcmoveind]][Fctourind[1][Fcmoveind]][Fctourind[2][Fcmoveind]] + Fcmove;

						for (int ii = 0; ii <= nhotind; ii++) {
							if (Particle[p].Fc[ii][Fctourind[1][Fcmoveind]][Fctourind[2][Fcmoveind]] > (1 - Frmin)) {
								Particle[p].Fc[ii][Fctourind[1][Fcmoveind]][Fctourind[2][Fcmoveind]] = 1;
							}
							if (Particle[p].Fc[ii][Fctourind[1][Fcmoveind]][Fctourind[2][Fcmoveind]] < Frmin) {
								Particle[p].Fc[ii][Fctourind[1][Fcmoveind]][Fctourind[2][Fcmoveind]] = 0;
							}
						}

						Fcsum = 0;
						for (int ii = 0; ii <= nhotind; ii++) {
							Fcsum = Fcsum + Particle[p].Fc[ii][Fctourind[1][Fcmoveind]][Fctourind[2][Fcmoveind]];
						}

						while (abs(1 - Fcsum) > Frmin) {
							x = 0;
							int ii = 0;
							Fctourind2[0].clear(); Fctourind2[1].clear(); Fctourind2[2].clear();
							for (x = 0; x <= ncsplits - 1; x++) {
								if (Fctourind[1][x] == Fctourind[1][Fcmoveind] && Fctourind[2][x] == Fctourind[2][Fcmoveind]) {
									Fctourind2[0].push_back(Fctourind[0][x]);
									Fctourind2[1].push_back(Fctourind[1][x]);
									Fctourind2[2].push_back(Fctourind[2][x]);
									ii++;
								}
							}
							//fixFCRI fixfcri(0, ii - 1);
							Fcmoveind = rand() % ii;// fixfcri(g);
							Particle[p].Fc[Fctourind2[0][Fcmoveind]][Fctourind2[1][Fcmoveind]][Fctourind2[2][Fcmoveind]] = Particle[p].Fc[Fctourind2[0][Fcmoveind]][Fctourind2[1][Fcmoveind]][Fctourind2[2][Fcmoveind]] - (Fcsum - 1);//Fcmove;

							//		Fcsum = 0;
							//		for (ii = 0; ii <= nhotind; ii++) {
							//			Fcsum = Fcsum + Particle[p].Fc[ii][Fctourind2[1][Fcmoveind]][Fctourind2[2][Fcmoveind]];
							//		}
							for (ii = 0; ii <= nhotind; ii++) {
								if (Particle[p].Fc[ii][Fctourind2[1][Fcmoveind]][Fctourind2[2][Fcmoveind]] > (1 - Frmin)) {
									Particle[p].Fc[ii][Fctourind2[1][Fcmoveind]][Fctourind2[2][Fcmoveind]] = 1;
								}
								if (Particle[p].Fc[ii][Fctourind2[1][Fcmoveind]][Fctourind2[2][Fcmoveind]] < Frmin) {
									Particle[p].Fc[ii][Fctourind2[1][Fcmoveind]][Fctourind2[2][Fcmoveind]] = 0;
									Particle[p].Q[ii][Fctourind2[1][Fcmoveind]][Fctourind2[2][Fcmoveind]] = 0;
								}
							}
							Fcsum = 0;
							for (ii = 0; ii <= nhotind; ii++) {
								Fcsum = Fcsum + Particle[p].Fc[ii][Fctourind2[1][Fcmoveind]][Fctourind2[2][Fcmoveind]];
							}
						}
					}


					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								if (Particle[p].Q[i][j][k] < Qmin || Particle[p].Fh[i][j][k] < Frmin || Particle[p].Fc[i][j][k] < Frmin) {
									Particle[p].Q[i][j][k] = 0;
									//Particle[p].Fh[i][j][k] = 0;
									//Particle[p].Fc[i][j][k] = 0;
								}
							}
						}
					}

					for (k = 0; k <= nstagesind; k++) {

						for (i = 0; i <= nhotind; i++) {
							Fhtourind2[0].clear(); Fhtourind2[1].clear(); Fhtourind2[2].clear();
							Fhsum = 0;
							jj = 0;
							for (j = 0; j <= ncoldind; j++) {

								if (varybin[i][j][k] == 1) {
									//if (Particle[p].Q[i][j][k] > Qmin) {
									Fhtourind2[0].push_back(i);
									Fhtourind2[1].push_back(j);
									Fhtourind2[2].push_back(k);
									jj++;
									//}
									//else {
										//Particle[p].Fh[i][j][k] = 0;
									//}
								}
								Fhsum = Fhsum + Particle[p].Fh[i][j][k];
							}
							if (jj > 0) {
								while (abs(1 - Fhsum) > Frmin) {
									remfracfrom = rand() % jj;
									Particle[p].Fh[Fhtourind2[0][remfracfrom]][Fhtourind2[1][remfracfrom]][Fhtourind2[2][remfracfrom]] = Particle[p].Fh[Fhtourind2[0][remfracfrom]][Fhtourind2[1][remfracfrom]][Fhtourind2[2][remfracfrom]] + (1 - Fhsum);
									Fhsum = 0;
									for (j = 0; j <= ncoldind; j++) {
										if (varybin[i][j][k] == 1) {
											if (Particle[p].Fh[i][j][k] > (1 - Frmin)) {
												Particle[p].Fh[i][j][k] = 1;
											}
											if (Particle[p].Fh[i][j][k] < Frmin) {
												Particle[p].Fh[i][j][k] = 0;
											}

										}
										Fhsum = Fhsum + Particle[p].Fh[i][j][k];
									}
								}
							}
							/*
							if (jj > 1 && Fhsum != 1) {
							remfracfrom = rand() % jj;
							Particle[p].Fh[Fhtourind2[0][remfracfrom]][Fhtourind2[1][remfracfrom]][Fhtourind2[2][remfracfrom]] = Particle[p].Fh[Fhtourind2[0][remfracfrom]][Fhtourind2[1][remfracfrom]][Fhtourind2[2][remfracfrom]] + (1 - Fhsum);
							}
							*/
						}

					}

					for (k = 0; k <= nstagesind; k++) {

						for (j = 0; j <= ncoldind; j++) {
							Fctourind2[0].clear(); Fctourind2[1].clear(); Fctourind2[2].clear();
							Fcsum = 0;
							ii = 0;
							for (i = 0; i <= nhotind; i++) {

								if (varybin[i][j][k] == 1) {
									//if (Particle[p].Q[i][j][k] > Qmin) {
									Fctourind2[0].push_back(i);
									Fctourind2[1].push_back(j);
									Fctourind2[2].push_back(k);
									ii++;
									//}
									//else {
										//Particle[p].Fc[i][j][k] = 0;
									//}
									Fcsum = Fcsum + Particle[p].Fc[i][j][k];
								}
							}
							if (ii > 0) {
								if (ii > 1) {
									int aaa = 1;
								}
								while (abs(1 - Fcsum) > Frmin) {
									remfracfrom = rand() % ii;
									Particle[p].Fc[Fctourind2[0][remfracfrom]][Fctourind2[1][remfracfrom]][Fctourind2[2][remfracfrom]] = Particle[p].Fc[Fctourind2[0][remfracfrom]][Fctourind2[1][remfracfrom]][Fctourind2[2][remfracfrom]] + (1 - Fcsum);
									Fcsum = 0;
									for (i = 0; i <= nhotind; i++) {
										if (varybin[i][j][k] == 1) {
											if (Particle[p].Fc[i][j][k] > (1 - Frmin)) {
												Particle[p].Fc[i][j][k] = 1;
											}
											if (Particle[p].Fc[i][j][k] < Frmin) {
												Particle[p].Fc[i][j][k] = 0;
											}

										}
										Fcsum = Fcsum + Particle[p].Fc[i][j][k];
									}
								}
							}
							/*
							if (ii > 1 && Fcsum != 1) {
							remfracfrom = rand() % ii;
							Particle[p].Fc[Fctourind2[0][remfracfrom]][Fctourind2[1][remfracfrom]][Fctourind2[2][remfracfrom]] = Particle[p].Fc[Fctourind2[0][remfracfrom]][Fctourind2[1][remfracfrom]][Fctourind2[2][remfracfrom]] + (1 - Fcsum);
							}
							*/
						}

					}

					if (reuseflag == 1) {
						Particle[0].Q = sol.Q;
						Particle[0].Fc = sol.Fc;
						Particle[0].Fh = sol.Fh;
						Particle[0].TotalCosts = sol.TotalCosts;
						Particle[0].TPenCosts = sol.TPenCosts;
						//memcpy(&Particle[0], &sol, sizeof(sol));
						printf_s("\n========= REUSING sol. ========= ");
						reuseflag = 0;
					}

					if (inputsol == 1 && firstcalc == 1 && firstsol == 1) {
						firstsol = 0;
						Particle[0] = HENSolution;
					}
					SWS(spaghetti, nstages, CaseStudyHEN, Particle[0], HSI, 0);

					//================================================================
					if (firstcalc == 1) {

						firstcalc = 0;
						solContVars = Particle[0];
						bestsolCont = Particle[0];
						//memcpy(&solContVars, &Particle[0], sizeof(Particle[0]));
						//memcpy(&bestsolCont, &Particle[0], sizeof(Particle[0]));


						for (k = 0; k <= nstagesind; k++) {
							for (j = 0; j <= ncoldind; j++) {
								for (i = 0; i <= nhotind; i++) {
									if (NewSol.z[i][j][k] == 1) {
										Qmaxm1[i][j][k] = Qmax[i][j][k];
									}
								}
							}
						}
					}
					else {
						if (Particle[0].TotalCosts < solContVars.TotalCosts) {
							solContVars = Particle[0];
							//memcpy(&solContVars, &Particle[0], sizeof(Particle[0]));

							for (k = 0; k <= nstagesind; k++) {
								for (j = 0; j <= ncoldind; j++) {
									for (i = 0; i <= nhotind; i++) {
										if (NewSol.z[i][j][k] == 1) {
											Qmaxm1[i][j][k] = Qmax[i][j][k];
										}
									}
								}
							}
						}
						else {
							double crr = (double)rand() / RAND_MAX;
							double cdeltasol = Particle[0].TotalCosts - solContVars.TotalCosts;
							if (crr < exp(-cdeltasol / cT)) {
								solContVars = Particle[0];
								//memcpy(&solContVars, &Particle[0], sizeof(Particle[0]));

								for (k = 0; k <= nstagesind; k++) {
									for (j = 0; j <= ncoldind; j++) {
										for (i = 0; i <= nhotind; i++) {
											if (NewSol.z[i][j][k] == 1) {
												Qmaxm1[i][j][k] = Qmax[i][j][k];
											}
										}
									}
								}
							}
							else {
								for (k = 0; k <= nstagesind; k++) {
									for (j = 0; j <= ncoldind; j++) {
										for (i = 0; i <= nhotind; i++) {
											if (NewSol.z[i][j][k] == 1) {
												Qmax[i][j][k] = Qmaxm1[i][j][k];
											}
										}
									}
								}
							}
							Particle[0] = solContVars;
							//memcpy(&Particle[0], &solContVars, sizeof(solContVars));
						}
					}
					//printf("Current - $%.2f | T = %.2f \n", Particle[0].TotalCosts, cT);
					if (Particle[0].TotalCosts < bestsolCont.TotalCosts) {
						bestsolCont = Particle[0];
						//memcpy(&bestsolCont, &Particle[0], sizeof(Particle[0]));
						//printf("Best - $%.2f | T = %.2f \n", bestsolCont.TotalCosts, cT);
					}
					//AreaCosts = 0;
					//UtilCosts = 0;
					//TPenCosts = 0;
					//TotalQhu = 0;
					//TotalQcu = 0;
					contmove = 0;
					Fhmove = 0;
					Fcmove = 0;
					//for (k = 0; k <= nstagesind; k++) {
					//	for (j = 0; j <= ncoldind; j++) {
					//		for (i = 0; i <= nhotind; i++) {
					//			if (z[i][j][k] == 1) {
					//				Area[i][j][k] = 0;
					//				LMTD[i][j][k] = 0;
					//			}
					//		}
					//	}
					//}

				}

				cT = cT * RFOParam.calpha;
				//std::cout << cT << endl;
			}
			printf("SA Best - $%.2f | T = %.2f | ", bestsolCont.TotalCosts, Tk);

			//===============================================================================
			//=== PARTICLE SWARM OPTIMIZATION ===============================================
			//===============================================================================
			Particle[0] = bestsolCont;
			//memcpy(&Particle[0], &bestsolCont, sizeof(bestsolCont));

			for (k = 0; k <= nstagesind; k++) {
				for (j = 0; j <= ncoldind; j++) {
					for (i = 0; i <= nhotind; i++) {
						if (NewSol.z[i][j][k] == 1) {
							Qmax[i][j][k] = Qmax0[i][j][k];
						}
					}
				}
			}

			for (k = 0; k <= nstagesind; k++) {
				for (i = 0; i <= nhotind; i++) {
					for (j = 0; j <= ncoldind; j++) {
						if (varybin[i][j][k] == 1) {
							Particle[0].VelQ[i][j][k] = RFOParam.v0factor * (double)rand() / (double)RAND_MAX * Qmax0[i][j][k];
						}
						else {
							Particle[0].VelQ[i][j][k] = 0;
						}
					}
				}
			}

			x = 0;
			for (k = 0; k <= nstagesind; k++) {
				for (i = 0; i <= nhotind; i++) {
					x = 0;
					for (j = 0; j <= ncoldind; j++) {
						if (varybin[i][j][k] == 1 && Fhtourind[0].size() > 0) {
							if (i == Fhtourind[0][x] && j == Fhtourind[1][x] && k == Fhtourind[2][x]) {
								Particle[0].VelFh[i][j][k] = RFOParam.v0ffactor * (2 * (double)rand() / (double)RAND_MAX - 1);
								x++;
							}
							else {
								Particle[0].VelFh[i][j][k] = 0;
							}
						}
						else {
							Particle[0].VelFh[i][j][k] = 0;
						}
					}
				}
			}
			x = 0;
			for (k = 0; k <= nstagesind; k++) {
				for (i = 0; i <= nhotind; i++) {
					x = 0;
					for (j = 0; j <= ncoldind; j++) {
						if (varybin[i][j][k] == 1 && Fctourind[0].size() > 0) {
							if (varybin[i][j][k] == 1 && i == Fctourind[0][x] && j == Fctourind[1][x] && k == Fctourind[2][x]) {
								Particle[0].VelFc[i][j][k] = (2 * (double)rand() / (double)RAND_MAX - 1) * RFOParam.v0ffactor;
								x++;
							}
							else {
								Particle[0].VelFc[i][j][k] = 0;
							}
						}
						else {
							Particle[0].VelFc[i][j][k] = 0;
						}
					}
				}
			}

			//===============================================================================
			//=== RANDOM PARTICLES GENERATION ===============================================
			//===============================================================================
			ParticleBest[0] = Particle[0];
			//memcpy(&ParticleBest[0], &Particle[0], sizeof(Particle[0]));

			int GlobalBestind = 0;
			GlobalBest = ParticleBest[0];
			//memcpy(&GlobalBest, &ParticleBest[0], sizeof(ParticleBest[0]));

			for (p = 1; p <= Particles - 1; p++) {

				for (k = 0; k <= nstagesind; k++) {
					for (i = 0; i <= nhotind; i++) {
						for (j = 0; j <= ncoldind; j++) {
							if (NewSol.z[i][j][k] == 1) {
								if (varybin[i][j][k] == 1) {
									Particle[p].Q[i][j][k] = (double)rand() / (double)RAND_MAX * Qmax0[i][j][k];
								}
								else {
									Particle[p].Q[i][j][k] = sol.Q[i][j][k];
								}
							}
						}
					}
				}

				//Heat Load Constraint Handling
				//Qhk = Qh - sum(sum(Particle(p).Q, 3), 2);
				double Qhkaux = 0;
				for (i = 0; i <= nhotind; i++) {
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							if (Particle[p].Q[i][j][k] > Qmin || Particle[p].Q[i][j][k] < -Qmin) {
								Qhkaux = Qhkaux + Particle[p].Q[i][j][k];
							};
						};
					};
					CaseStudyHEN.Qhk[i] = CaseStudyHEN.Qh[i] - Qhkaux;
					Qhkaux = 0;
				};


				//Qck = Qc - sum(sum(Particle(p).Q, 3), 1)';
				double Qckaux = 0;
				for (j = 0; j <= ncoldind; j++) {
					for (k = 0; k <= nstagesind; k++) {
						for (i = 0; i <= nhotind; i++) {
							if (Particle[p].Q[i][j][k] > Qmin || Particle[p].Q[i][j][k] < -Qmin) {
								Qckaux = Qckaux + Particle[p].Q[i][j][k];
							};
						};
					};
					CaseStudyHEN.Qck[j] = CaseStudyHEN.Qc[j] - Qckaux;
					Qckaux = 0;
				};

				for (k = 0; k <= nstagesind; k++) {
					for (j = 0; j <= ncoldind; j++) {
						for (i = 0; i <= nhotind; i++) {
							if (NewSol.z[i][j][k] == 1) {
								if (CaseStudyHEN.Qhk[i] > CaseStudyHEN.Qck[j]) {
									Qmax[i][j][k] = CaseStudyHEN.Qck[j];
								}
								else {
									Qmax[i][j][k] = CaseStudyHEN.Qhk[i];
								}
							}
						}
					}
				}

				//Check if some Qmax is < 0
				int QmaxnotOKsum = 0;
				vector<vector<int>> QmaxnotOK; QmaxnotOK.resize(3);
				for (k = 0; k <= nstagesind; k++) {
					for (j = 0; j <= ncoldind; j++) {
						for (i = 0; i <= nhotind; i++) {
							if (Qmax[i][j][k] < 0 && Particle[p].Q[i][j][k] > 0) {
								QmaxnotOK[0].push_back(i);
								QmaxnotOK[1].push_back(j);
								QmaxnotOK[2].push_back(k);
								QmaxnotOKsum = QmaxnotOKsum + 1;
							}
						}
					}
				}
				while (QmaxnotOKsum > 0) {
					//fixRI fixri(0, QmaxnotOKsum);
					int indtofix = rand() % QmaxnotOKsum;//fixri(g);

					Particle[p].Q[QmaxnotOK[0][indtofix]][QmaxnotOK[1][indtofix]][QmaxnotOK[2][indtofix]] = Particle[p].Q[QmaxnotOK[0][indtofix]][QmaxnotOK[1][indtofix]][QmaxnotOK[2][indtofix]] + Qmax[QmaxnotOK[0][indtofix]][QmaxnotOK[1][indtofix]][QmaxnotOK[2][indtofix]];

					//Qhk = Qh - sum(sum(Particle(p).Q, 3), 2);
					Qhkaux = 0;
					for (i = 0; i <= nhotind; i++) {
						for (k = 0; k <= nstagesind; k++) {
							for (j = 0; j <= ncoldind; j++) {
								if (Particle[p].Q[i][j][k] > Qmin) {
									Qhkaux = Qhkaux + Particle[p].Q[i][j][k];
								};
							};
						};
						CaseStudyHEN.Qhk[i] = CaseStudyHEN.Qh[i] - Qhkaux;
						Qhkaux = 0;
					};


					//Qck = Qc - sum(sum(Particle(p).Q, 3), 1)';
					for (j = 0; j <= ncoldind; j++) {
						for (k = 0; k <= nstagesind; k++) {
							for (i = 0; i <= nhotind; i++) {
								if (Particle[p].Q[i][j][k] > Qmin) {
									Qckaux = Qckaux + Particle[p].Q[i][j][k];
								};
							};
						};
						CaseStudyHEN.Qck[j] = CaseStudyHEN.Qc[j] - Qckaux;
						Qckaux = 0;
					};

					//Update Qmax
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								if (NewSol.z[i][j][k] == 1) {
									if (CaseStudyHEN.Qhk[i] > CaseStudyHEN.Qck[j]) {
										Qmax[i][j][k] = CaseStudyHEN.Qck[j];
									}
									else {
										Qmax[i][j][k] = CaseStudyHEN.Qhk[i];
									}
								}
							}
						}
					}

					//Check if some Qmax is < 0
					QmaxnotOKsum = 0;
					QmaxnotOK[0].clear(); QmaxnotOK[1].clear(); QmaxnotOK[2].clear();
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								if (Qmax[i][j][k] < 0 && Particle[p].Q[i][j][k] > Qmin) {
									QmaxnotOK[0].push_back(i);
									QmaxnotOK[1].push_back(j);
									QmaxnotOK[2].push_back(k);
									QmaxnotOKsum = QmaxnotOKsum + 1;
								}
							}
						}
					}
				}

				// Heat Loads now must be within feasible region

				//Initializing Fractions
				//Hot Stream Splits
				double nhsplitsaux = 0;
				for (k = 0; k <= nstagesind; k++) {
					for (i = 0; i <= nhotind; i++) {
						Fhsum = 0;
						for (j = 0; j <= ncoldind; j++) {
							if (NewSol.z[i][j][k] == 1) {
								nhsplitsaux = nhsplitsaux + 1;
							}
						}
						if (nhsplitsaux > 1) {
							for (j = 0; j <= ncoldind; j++) {
								if (NewSol.z[i][j][k] == 1) {
									if (varybin[i][j][k] == 1) {
										Particle[p].Fh[i][j][k] = (double)rand() / (double)RAND_MAX;
										Fhsum = Fhsum + Particle[p].Fh[i][j][k];
									}
									else {
										Particle[p].Fh[i][j][k] = sol.Fh[i][j][k];
									}
								}
							}
							for (j = 0; j <= ncoldind; j++) {
								if (varybin[i][j][k] == 1) {
									Particle[p].Fh[i][j][k] = Particle[p].Fh[i][j][k] / Fhsum;
								}
							}
						}
						else {
							for (j = 0; j <= ncoldind; j++) {
								if (varybin[i][j][k] == 1) {
									Particle[p].Fh[i][j][k] = 1;
								}
							}
						}
						nhsplitsaux = 0;
					}
				}
				double ncsplitsaux = 0;
				Fcsum = 0;
				//Cold Stream Splits
				for (k = 0; k <= nstagesind; k++) {
					for (j = 0; j <= ncoldind; j++) {
						Fcsum = 0;
						for (i = 0; i <= nhotind; i++) {
							if (NewSol.z[i][j][k] == 1) {
								ncsplitsaux = ncsplitsaux + 1;
							}
						}
						if (ncsplitsaux > 1) {
							for (i = 0; i <= nhotind; i++) {
								if (NewSol.z[i][j][k] == 1) {
									if (varybin[i][j][k] == 1) {
										Particle[p].Fc[i][j][k] = (double)rand() / (double)RAND_MAX;
										Fcsum = Fcsum + Particle[p].Fc[i][j][k];
									}
									else {
										Particle[p].Fc[i][j][k] = sol.Fc[i][j][k];
									}
								}
							}
							for (i = 0; i <= nhotind; i++) {
								if (varybin[i][j][k] == 1) {
									Particle[p].Fc[i][j][k] = Particle[p].Fc[i][j][k] / Fcsum;
								}
							}
						}
						else {
							for (i = 0; i <= nhotind; i++) {
								if (varybin[i][j][k] == 1) {
									Particle[p].Fc[i][j][k] = 1;
								}
							}
						}
						ncsplitsaux = 0;
					}
				}

				//Assigning Particles with random velocities
				for (k = 0; k <= nstagesind; k++) {
					for (i = 0; i <= nhotind; i++) {
						for (j = 0; j <= ncoldind; j++) {
							if (varybin[i][j][k] == 1) {
								Particle[p].VelQ[i][j][k] = (2 * (double)rand() / (double)RAND_MAX - 1) * Qmax0[i][j][k] * RFOParam.v0factor;
							}
							else {
								Particle[p].VelQ[i][j][k] = 0;
							}
						}
					}
				}
				x = 0;
				for (k = 0; k <= nstagesind; k++) {
					for (i = 0; i <= nhotind; i++) {
						x = 0;
						for (j = 0; j <= ncoldind; j++) {
							if (varybin[i][j][k] == 1 && Fhtourind[0].size() > 0) {
								if (i == Fhtourind[0][x] && j == Fhtourind[1][x] && k == Fhtourind[2][x]) {
									Particle[p].VelFh[i][j][k] = RFOParam.v0ffactor * (2 * (double)rand() / (double)RAND_MAX - 1);
									x++;
								}
								else {
									Particle[p].VelFh[i][j][k] = 0;
								}
							}
							else {
								Particle[p].VelFh[i][j][k] = 0;
							}
						}
					}
				}
				x = 0;
				for (k = 0; k <= nstagesind; k++) {
					for (i = 0; i <= nhotind; i++) {
						x = 0;
						for (j = 0; j <= ncoldind; j++) {
							if (varybin[i][j][k] == 1 && Fctourind[0].size() > 0) {
								if (varybin[i][j][k] == 1 && i == Fctourind[0][x] && j == Fctourind[1][x] && k == Fctourind[2][x]) {
									Particle[p].VelFc[i][j][k] = (2 * (double)rand() / (double)RAND_MAX - 1) * RFOParam.v0ffactor;
									x++;
								}
								else {
									Particle[p].VelFc[i][j][k] = 0;
								}
							}
							else {
								Particle[p].VelFc[i][j][k] = 0;
							}
						}
					}
				}
				//=================================================
				//Objective function calculation
				//=================================================
				SWS(spaghetti, nstages, CaseStudyHEN, Particle[p], HSI, 0);

				ParticleBest[p] = Particle[p];
				//memcpy(&ParticleBest[p], &Particle[p], sizeof(Particle[p]));
				if (ParticleBest[p].TotalCosts < GlobalBest.TotalCosts) {
					GlobalBestind = p;
					GlobalBest = ParticleBest[p];// memcpy(&GlobalBest, &ParticleBest[p], sizeof(ParticleBest[p]));
				}
			}
			double wmax = RFOParam.wmax;
			double wmin = RFOParam.wmin;
			double c1 = RFOParam.c1;
			double c2 = RFOParam.c2;

			vector<vector<int>> QmaxnotOK; QmaxnotOK.resize(3);
			QmaxnotOK[0].resize(SAParam.maxHE); QmaxnotOK[1].resize(SAParam.maxHE); QmaxnotOK[2].resize(SAParam.maxHE);
			vector<vector<int>> Fhtourind2; Fhtourind2.resize(3);
			vector<vector<int>> Fctourind2; Fctourind2.resize(3);
			Fhtourind2[0].resize(SAParam.maxHE); Fhtourind2[1].resize(SAParam.maxHE); Fhtourind2[2].resize(SAParam.maxHE);
			Fctourind2[0].resize(SAParam.maxHE); Fctourind2[1].resize(SAParam.maxHE); Fctourind2[2].resize(SAParam.maxHE);

			for (int PSOiter = 0; PSOiter < RFOParam.PSOMaxIter; PSOiter++) {
				for (p = 0; p <= Particles - 1; p++) {
					double r1 = (double)rand() / (double)RAND_MAX;
					double r2 = (double)rand() / (double)RAND_MAX;
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								if (varybin[i][j][k] == 1) {
									Particle[p].VelQ[i][j][k] = ((wmax - wmin) * (RFOParam.PSOMaxIter - PSOiter) / RFOParam.PSOMaxIter + wmin) * Particle[p].VelQ[i][j][k] + c1 * r1 * (ParticleBest[p].Q[i][j][k] - Particle[p].Q[i][j][k]) + c2 * r2 * (GlobalBest.Q[i][j][k] - Particle[p].Q[i][j][k]);
									Particle[p].VelFh[i][j][k] = ((wmax - wmin) * (RFOParam.PSOMaxIter - PSOiter) / RFOParam.PSOMaxIter + wmin) * Particle[p].VelFh[i][j][k] + c1 * r1 * (ParticleBest[p].Fh[i][j][k] - Particle[p].Fh[i][j][k]) + c2 * r2 * (GlobalBest.Fh[i][j][k] - Particle[p].Fh[i][j][k]);
									Particle[p].VelFc[i][j][k] = ((wmax - wmin) * (RFOParam.PSOMaxIter - PSOiter) / RFOParam.PSOMaxIter + wmin) * Particle[p].VelFc[i][j][k] + c1 * r1 * (ParticleBest[p].Fc[i][j][k] - Particle[p].Fc[i][j][k]) + c2 * r2 * (GlobalBest.Fc[i][j][k] - Particle[p].Fc[i][j][k]);

									Particle[p].Q[i][j][k] = Particle[p].Q[i][j][k] + Particle[p].VelQ[i][j][k];
									Particle[p].Fh[i][j][k] = Particle[p].Fh[i][j][k] + Particle[p].VelFh[i][j][k];
									Particle[p].Fc[i][j][k] = Particle[p].Fc[i][j][k] + Particle[p].VelFc[i][j][k];

									if (Particle[p].Q[i][j][k] < Qmin || Particle[p].Fh[i][j][k] < Frmin || Particle[p].Fc[i][j][k] < Frmin) {
										Particle[p].Q[i][j][k] = 0;
										Particle[p].Fh[i][j][k] = 0;
										Particle[p].Fc[i][j][k] = 0;
									}

								}
							}
						}
					}

					//Heat Load Constraint Handling
					//Qhk = Qh - sum(sum(Particle(p).Q, 3), 2);
					double Qhkaux = 0;
					for (i = 0; i <= nhotind; i++) {
						for (k = 0; k <= nstagesind; k++) {
							for (j = 0; j <= ncoldind; j++) {
								if (Particle[p].Q[i][j][k] > Qmin || Particle[p].Q[i][j][k] < -Qmin) {
									Qhkaux = Qhkaux + Particle[p].Q[i][j][k];
								};
							};
						};
						CaseStudyHEN.Qhk[i] = CaseStudyHEN.Qh[i] - Qhkaux;
						Qhkaux = 0;
					};


					//Qck = Qc - sum(sum(Particle(p).Q, 3), 1)';
					double Qckaux = 0;
					for (j = 0; j <= ncoldind; j++) {
						for (k = 0; k <= nstagesind; k++) {
							for (i = 0; i <= nhotind; i++) {
								if (Particle[p].Q[i][j][k] > Qmin || Particle[p].Q[i][j][k] < -Qmin) {
									Qckaux = Qckaux + Particle[p].Q[i][j][k];
								};
							};
						};
						CaseStudyHEN.Qck[j] = CaseStudyHEN.Qc[j] - Qckaux;
						Qckaux = 0;
					};

					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								if (NewSol.z[i][j][k] == 1) {
									if (CaseStudyHEN.Qhk[i] > CaseStudyHEN.Qck[j]) {
										Qmax[i][j][k] = CaseStudyHEN.Qck[j];
									}
									else {
										Qmax[i][j][k] = CaseStudyHEN.Qhk[i];
									}
								}
							}
						}
					}

					//Check if some Qmax is < 0
					int QmaxnotOKsum = 0;
					//vector<vector<int>> QmaxnotOK; QmaxnotOK.resize(3);
					//QmaxnotOK[0].clear(); QmaxnotOK[1].clear(); QmaxnotOK[2].clear();
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								if (Qmax[i][j][k] < 0 && Particle[p].Q[i][j][k] > 0) {
									QmaxnotOK[0][QmaxnotOKsum] = i;
									QmaxnotOK[1][QmaxnotOKsum] = j;
									QmaxnotOK[2][QmaxnotOKsum] = k;
									QmaxnotOKsum = QmaxnotOKsum + 1;
								}
							}
						}
					}
					while (QmaxnotOKsum > 0) {
						//fixRI fixri(0, QmaxnotOKsum);
						int indtofix = rand() % QmaxnotOKsum;//fixri(g);

						Particle[p].Q[QmaxnotOK[0][indtofix]][QmaxnotOK[1][indtofix]][QmaxnotOK[2][indtofix]] = Particle[p].Q[QmaxnotOK[0][indtofix]][QmaxnotOK[1][indtofix]][QmaxnotOK[2][indtofix]] + Qmax[QmaxnotOK[0][indtofix]][QmaxnotOK[1][indtofix]][QmaxnotOK[2][indtofix]];

						//Qhk = Qh - sum(sum(Particle(p).Q, 3), 2);
						Qhkaux = 0;
						for (i = 0; i <= nhotind; i++) {
							for (k = 0; k <= nstagesind; k++) {
								for (j = 0; j <= ncoldind; j++) {
									if (Particle[p].Q[i][j][k] > Qmin) {
										Qhkaux = Qhkaux + Particle[p].Q[i][j][k];
									};
								};
							};
							CaseStudyHEN.Qhk[i] = CaseStudyHEN.Qh[i] - Qhkaux;
							Qhkaux = 0;
						};

						//Qck = Qc - sum(sum(Particle(p).Q, 3), 1)';
						for (j = 0; j <= ncoldind; j++) {
							for (k = 0; k <= nstagesind; k++) {
								for (i = 0; i <= nhotind; i++) {
									if (Particle[p].Q[i][j][k] > Qmin) {
										Qckaux = Qckaux + Particle[p].Q[i][j][k];
									};
								};
							};
							CaseStudyHEN.Qck[j] = CaseStudyHEN.Qc[j] - Qckaux;
							Qckaux = 0;
						};

						//Update Qmax
						for (k = 0; k <= nstagesind; k++) {
							for (j = 0; j <= ncoldind; j++) {
								for (i = 0; i <= nhotind; i++) {
									if (NewSol.z[i][j][k] == 1) {
										if (CaseStudyHEN.Qhk[i] > CaseStudyHEN.Qck[j]) {
											Qmax[i][j][k] = CaseStudyHEN.Qck[j];
										}
										else {
											Qmax[i][j][k] = CaseStudyHEN.Qhk[i];
										}
									}
								}
							}
						}

						//Check if some Qmax is < 0
						QmaxnotOKsum = 0;
						//QmaxnotOK[0].clear(); QmaxnotOK[1].clear(); QmaxnotOK[2].clear();
						;						for (k = 0; k <= nstagesind; k++) {
							for (j = 0; j <= ncoldind; j++) {
								for (i = 0; i <= nhotind; i++) {
									if (Qmax[i][j][k] < 0 && Particle[p].Q[i][j][k] > Qmin) {
										QmaxnotOK[0][QmaxnotOKsum] = i;
										QmaxnotOK[1][QmaxnotOKsum] = j;
										QmaxnotOK[2][QmaxnotOKsum] = k;
										QmaxnotOKsum = QmaxnotOKsum + 1;
									}
								}
							}
						}
					}

					// Particles now must be within feasible region
					//vector<vector<int>> Fhtourind2; Fhtourind2.resize(3);
					//vector<vector<int>> Fctourind2; Fctourind2.resize(3);
					//Fhtourind2[0].clear(); Fhtourind2[1].clear(); Fhtourind2[2].clear();
					//Fctourind2[0].clear(); Fctourind2[1].clear(); Fctourind2[2].clear();
					for (k = 0; k <= nstagesind; k++) {

						for (i = 0; i <= nhotind; i++) {
							//Fhtourind2[0].clear(); Fhtourind2[1].clear(); Fhtourind2[2].clear();
							Fhsum = 0;
							int jj = 0;
							for (j = 0; j <= ncoldind; j++) {
								if (varybin[i][j][k] == 1) {
									//if (Particle[p].Q[i][j][k] > Qmin) {
									Fhtourind2[0][jj] = i;
									Fhtourind2[1][jj] = j;
									Fhtourind2[2][jj] = k;
									jj++;
									//}
									//else {
									//	Particle[p].Fh[i][j][k] = 0;
									//}
								}
								Fhsum = Fhsum + Particle[p].Fh[i][j][k];
							}
							if (jj > 0) {
								while (abs(1 - Fhsum) > Frmin) {
									int remfracfrom = rand() % jj;
									Particle[p].Fh[Fhtourind2[0][remfracfrom]][Fhtourind2[1][remfracfrom]][Fhtourind2[2][remfracfrom]] = Particle[p].Fh[Fhtourind2[0][remfracfrom]][Fhtourind2[1][remfracfrom]][Fhtourind2[2][remfracfrom]] + (1 - Fhsum);
									Fhsum = 0;
									for (j = 0; j <= ncoldind; j++) {
										if (varybin[i][j][k] == 1) {
											if (Particle[p].Fh[i][j][k] > (1 - Frmin)) {
												Particle[p].Fh[i][j][k] = 1;
											}
											if (Particle[p].Fh[i][j][k] < Frmin) {
												Particle[p].Fh[i][j][k] = 0;
											}

										}
										Fhsum = Fhsum + Particle[p].Fh[i][j][k];
									}
								}
							}
							/*
							if (jj > 1 && Fhsum != 1) {
							remfracfrom = rand() % jj;
							Particle[p].Fh[Fhtourind2[0][remfracfrom]][Fhtourind2[1][remfracfrom]][Fhtourind2[2][remfracfrom]] = Particle[p].Fh[Fhtourind2[0][remfracfrom]][Fhtourind2[1][remfracfrom]][Fhtourind2[2][remfracfrom]] + (1 - Fhsum);
							}
							*/
						}

					}

					for (k = 0; k <= nstagesind; k++) {

						for (j = 0; j <= ncoldind; j++) {
							//Fctourind2[0].clear(); Fctourind2[1].clear(); Fctourind2[2].clear();
							Fcsum = 0;
							int ii = 0;
							for (i = 0; i <= nhotind; i++) {
								if (varybin[i][j][k] == 1) {
									//if (Particle[p].Q[i][j][k] > Qmin) {
									Fctourind2[0][ii] = i;
									Fctourind2[1][ii] = j;
									Fctourind2[2][ii] = k;
									ii++;
									//}
									//else {
									//	Particle[p].Fc[i][j][k] = 0;
									//}
								}
								Fcsum = Fcsum + Particle[p].Fc[i][j][k];
							}
							if (ii > 0) {
								if (ii > 1) {
									int aaa = 1;
								}
								while (abs(1 - Fcsum) > Frmin) {
									int remfracfrom = rand() % ii;
									Particle[p].Fc[Fctourind2[0][remfracfrom]][Fctourind2[1][remfracfrom]][Fctourind2[2][remfracfrom]] = Particle[p].Fc[Fctourind2[0][remfracfrom]][Fctourind2[1][remfracfrom]][Fctourind2[2][remfracfrom]] + (1 - Fcsum);
									Fcsum = 0;
									for (i = 0; i <= nhotind; i++) {
										if (varybin[i][j][k] == 1) {
											if (Particle[p].Fc[i][j][k] > (1 - Frmin)) {
												Particle[p].Fc[i][j][k] = 1;
											}
											if (Particle[p].Fc[i][j][k] < Frmin) {
												Particle[p].Fc[i][j][k] = 0;
											}

										}
										Fcsum = Fcsum + Particle[p].Fc[i][j][k];
									}
								}
							}
							/*
							if (ii > 1 && Fcsum != 1) {
							remfracfrom = rand() % ii;
							Particle[p].Fc[Fctourind2[0][remfracfrom]][Fctourind2[1][remfracfrom]][Fctourind2[2][remfracfrom]] = Particle[p].Fc[Fctourind2[0][remfracfrom]][Fctourind2[1][remfracfrom]][Fctourind2[2][remfracfrom]] + (1 - Fcsum);
							}
							*/
						}

					}

					//=== Objective Function Calculation =============================
					SWS(spaghetti, nstages, CaseStudyHEN, Particle[p], HSI, 0);

					if (Particle[p].TotalCosts < ParticleBest[p].TotalCosts) {
						ParticleBest[p] = Particle[p];
						//memcpy(&ParticleBest[p], &Particle[p], sizeof(Particle[p]));
					}
					if (ParticleBest[p].TotalCosts < GlobalBest.TotalCosts) {
						GlobalBestind = p;

						/*

												for (k = 0; k <= nstagesind; k++) {
							for (j = 0; j <= ncoldind; j++) {
								for (i = 0; i <= nhotind; i++) {
									gbestThout[i][j][k] = Thout[i][j][k];
									gbestTcout[i][j][k] = Tcout[i][j][k];
									gbestLMTD[i][j][k] = LMTD[i][j][k];
									gbestArea[i][j][k] = Area[i][j][k];
								}
							}
						}
						for (k = 0; k <= nstagesind; k++) {
							for (j = 0; j <= ncoldind; j++) {
								gbestTck[j][k] = Tck[j][k];
							}
							for (i = 0; i <= nhotind; i++) {
								gbestThk[i][k] = Thk[i][k];
							}
						}

						for (j = 0; j <= ncoldind; j++) {
							gbestTcfinal0[j] = Tcfinal0[j];
							gbestTcfinal[j] = Tcfinal[j];
							gbestLMTDhu[j] = LMTDhu[j];
							gbestAreahu[j] = Areahu[j];
						}
						for (i = 0; i <= nhotind; i++) {
							gbestThfinal0[i] = Thfinal0[i];
							gbestThfinal[i] = Thfinal[i];
							gbestLMTDcu[i] = LMTDcu[i];
							gbestAreacu[i] = Areacu[i];
						}

						*/


					}
					/*

										for (k = 0; k <= nstagesind; k++) {
						for (i = 0; i <= nhotind; i++) {
							sumQThk[i][k] = 0;
						}
					}
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							sumQTck[j][k] = 0;
						}
					}
					AreaCosts = 0;
					UtilCosts = 0;
					TPenCosts = 0;
					TotalQhu = 0;
					TotalQcu = 0;
					contmove = 0;
					Fhmove = 0;
					Fcmove = 0;

					*/

					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								if (NewSol.z[i][j][k] == 1) {
									//Area[i][j][k] = 0;
									//LMTD[i][j][k] = 0;
									Qmax[i][j][k] = Qmax0[i][j][k];
								}
							}
						}
					}

				}
				GlobalBest = ParticleBest[GlobalBestind];
				//memcpy(&GlobalBest, &ParticleBest[GlobalBestind], sizeof(ParticleBest[GlobalBestind]));
				//printf("PSO Best - $%.6f | T = %.2f \n", GlobalBest.TotalCosts, PSOiter);


			}
			printf("PSO Best - $%.6f | iter = %.2f \n", GlobalBest.TotalCosts, RFOParam.PSOMaxIter);

			SWS(spaghetti, nstages, CaseStudyHEN, GlobalBest, HSI, 0);

			if (GlobalBest.TotalCosts < 450000 && spaghetti == 0) {
				//SWS(spaghetti, nstages, CaseStudyHEN, GlobalBest, HSI, 1);
			}
			//system("CLS");



			char TACTIMEHX[200];
			double end = clock();
			double time = (double)(end - start) / CLOCKS_PER_SEC;
			sprintf_s(TACTIMEHX, "%.2f\t%.2f\t%i\t%.2f\t", GlobalBest.TotalCosts, time, (int)nhe, Tk);
			myfile2 << TACTIMEHX;

			if (firstsol == 1 && inputsol == 0) {
				firstsol = 0;
				sol = GlobalBest;
				bestsol = GlobalBest;
				NewSol = GlobalBest;
				//memcpy(&sol, &GlobalBest, sizeof(GlobalBest));
				//memcpy(&bestsol, &GlobalBest, sizeof(GlobalBest));
				if (GlobalBest.TPenCosts == 0) {
					hasfoundnonpen = 1;
				}
				nhe = 0;
				for (k = 0; k <= nstagesind; k++) {
					for (j = 0; j <= ncoldind; j++) {
						for (i = 0; i <= nhotind; i++) {
							NewSol.z[i][j][k] = 0;
							sol.z[i][j][k] = 0;
							//zm1[i][j][k] = 0;
							if (GlobalBest.Q[i][j][k] > Qmin) {
								NewSol.z[i][j][k] = 1;
								sol.z[i][j][k] = 1;
								//zm1[i][j][k] = 1;
								nhe++;
							}
						}
					}
				}

				//char TACTIMEHX[400];
				//end = clock();
				//time = (double)(end - start) / CLOCKS_PER_SEC;
				//sprintf_s(TACTIMEHX, "%.2f\t%.2f\t%i\t%.2f\n", sol.TotalCosts, time, (int)nhe, Tk);
				//myfile1 << TACTIMEHX;

				//sprintf_s(TACTIMEHX, "%.2f\n", bestsol.TotalCosts);
				//myfile1 << TACTIMEHX;

				tourlength = nhe;
				tourlengthind = nhe - 1;
			}
			else {
				if (GlobalBest.TotalCosts < sol.TotalCosts && ((GlobalBest.TPenCosts > 0 && hasfoundnonpen == 0) || (GlobalBest.TPenCosts == 0))) {
					//memcpy(&sol, &GlobalBest, sizeof(GlobalBest));
					sol = GlobalBest;
					NewSol = GlobalBest;
					if (GlobalBest.TPenCosts == 0) {
						hasfoundnonpen = 1;
					}
					nhe = 0;
					for (k = 0; k <= nstagesind; k++) {
						for (j = 0; j <= ncoldind; j++) {
							for (i = 0; i <= nhotind; i++) {
								NewSol.z[i][j][k] = 0;
								sol.z[i][j][k] = 0;
								//zm1[i][j][k] = 0;
								if (GlobalBest.Q[i][j][k] > Qmin) {
									NewSol.z[i][j][k] = 1;
									sol.z[i][j][k] = 1;
									//zm1[i][j][k] = 1;
									nhe++;
								}
							}
						}
					}

					char TACTIMEHX[200];
					//	end = clock();
					//	time = (double)(end - start) / CLOCKS_PER_SEC;
					//	sprintf_s(TACTIMEHX, "%.2f\t%.2f\t%i\t%.2f\n", sol.TotalCosts, time, (int)nhe, Tk);
					//	myfile1 << TACTIMEHX;

					tourlength = nhe;
					tourlengthind = nhe - 1;
				}
				else {
					double rr = (double)rand() / (double)RAND_MAX;
					double deltasol = GlobalBest.TotalCosts - sol.TotalCosts;
					if (rr < exp(-deltasol / Tk) && ((GlobalBest.TPenCosts > 0 && hasfoundnonpen == 0) || (GlobalBest.TPenCosts == 0))) {
						//memcpy(&sol, &GlobalBest, sizeof(GlobalBest));
						sol = GlobalBest;
						NewSol = GlobalBest;
						if (GlobalBest.TPenCosts == 0) {
							hasfoundnonpen = 1;
						}
						nhe = 0;
						for (k = 0; k <= nstagesind; k++) {
							for (j = 0; j <= ncoldind; j++) {
								for (i = 0; i <= nhotind; i++) {
									NewSol.z[i][j][k] = 0;
									sol.z[i][j][k] = 0;
									//zm1[i][j][k] = 0;
									if (GlobalBest.Q[i][j][k] > Qmin) {
										NewSol.z[i][j][k] = 1;
										sol.z[i][j][k] = 1;
										//zm1[i][j][k] = 1;
										nhe++;
									}
								}
							}
						}

						char TACTIMEHX[200];
						//	end = clock();
						//	time = (double)(end - start) / CLOCKS_PER_SEC;
							//sprintf_s(TACTIMEHX, "%.2f\t%.2f\t%i\t%.2f\n", sol.TotalCosts, time, (int)nhe, Tk);
							//myfile1 << TACTIMEHX;

						tourlength = nhe;
						tourlengthind = nhe - 1;
					}
					else {
						NewSol = sol;
						//z[randomh][randomc][randoms] = zm1[randomh][randomc][randoms];

						/*if (deletedr == 1) {
						z[drandomh][drandomc][drandoms] = zm1[drandomh][drandomc][drandoms];//z[drandomti][deletedr][drandoms] = zm1[drandomti][deletedr][drandoms];
						deletedr = 0;
						}
						*/
					}
				}
				//sollog(solnum)=newsol;
				//solnum=solnum+1;
			}
			//char TACTIMEHX2[200];


			if (GlobalBest.TotalCosts < bestsol.TotalCosts && ((GlobalBest.TPenCosts > 0 && hasfoundnonpen == 0) || (GlobalBest.TPenCosts == 0))) {
				//memcpy(&bestsol, &GlobalBest, sizeof(GlobalBest));
				bestsol = GlobalBest;
				end = clock();
				time = (double)(end - start) / CLOCKS_PER_SEC;
				bestsoltime = time;
				bestnhe = nhe;
				/*



				for (k = 0; k <= nstagesind; k++) {
					for (j = 0; j <= ncoldind; j++) {
						for (i = 0; i <= nhotind; i++) {
							bestThout[i][j][k] = gbestThout[i][j][k];
							bestTcout[i][j][k] = gbestTcout[i][j][k];
							bestLMTD[i][j][k] = gbestLMTD[i][j][k];
							bestArea[i][j][k] = gbestArea[i][j][k];
						}
					}
				}
				for (k = 0; k <= nstagesind; k++) {
					for (j = 0; j <= ncoldind; j++) {
						bestTck[j][k] = gbestTck[j][k];
					}
					for (i = 0; i <= nhotind; i++) {
						bestThk[i][k] = gbestThk[i][k];
					}
				}

				for (j = 0; j <= ncoldind; j++) {
					bestTcfinal0[j] = gbestTcfinal0[j];
					bestTcfinal[j] = gbestTcfinal[j];
					bestLMTDhu[j] = gbestLMTDhu[j];
					bestAreahu[j] = gbestAreahu[j];
				}
				for (i = 0; i <= nhotind; i++) {
					bestThfinal0[i] = gbestThfinal0[i];
					bestThfinal[i] = gbestThfinal[i];
					bestLMTDcu[i] = gbestLMTDcu[i];
					bestAreacu[i] = gbestAreacu[i];
				}

				bestTotalQcu = gbestTotalQcu;
				bestTotalQhu = gbestTotalQhu;
				*/

				nonacceptcont = 0;
				printf_s("Non-accept count = %i\n", nonacceptcont);
			}
			else {
				nonacceptcont++;
				printf_s("Non-accept count = %i\n", nonacceptcont);
			}

			if (spaghetti == 1) {
				double QCUTOT = 0;
				for (int i = 0; i <= bestsol.HSI.Qcu.size() - 1; i++) {
					if (isnan(bestsol.HSI.Qcu[i]) == 0) {
						QCUTOT = QCUTOT + bestsol.HSI.Qcu[i];
					}
				}
				double QHUTOT = 0;
				for (int i = 0; i <= bestsol.HSI.Qhu.size() - 1; i++) {
					if (isnan(bestsol.HSI.Qhu[i]) == 0) {
						QHUTOT = QHUTOT + bestsol.HSI.Qhu[i];
					}
				}
				if (QCUTOT < 0.0001 && QHUTOT < 0.0001) {
					endopt = 1;
					break;
				}

			}

			/*
			double sumqhuf = 0;
			double sumqcuf = 0;
			for (j = 0; j <= ncoldind; j++) {
				if (bestsol.Qhu[j] > Qmin) {
					sumqhuf = sumqhuf + bestsol.Qhu[j];
				}

			}
			for (i = 0; i <= nhotind; i++) {
				if (bestsol.Qcu[i] > Qmin) {
					sumqcuf = sumqcuf + bestsol.Qcu[i];
				}
			}

			if (sumqhuf == 0 && sumqcuf == 0) {
				utiliszero = 1;
			}
			else {
				utiliszero = 0;
			}
			if (ActivateSpaghetti == 1 && attcont < 20) {
				if (nonacceptcont == 10) {
					printf_s("\n=================\nReverting to bestsol.\n=================\n");
					memcpy(&sol, &bestsol, sizeof(bestsol));
					bestsolreuseflag = 1;
					nonacceptcont = 0;
				}
				else {
					bestsolreuseflag = 0;
				}
			}
			else {
				if (nonacceptcont == 50) {
					printf_s("\n=================\nReverting to bestsol.\n=================\n");
					memcpy(&sol, &bestsol, sizeof(bestsol));
					bestsolreuseflag = 1;
					nonacceptcont = 0;
				}
				else {
					bestsolreuseflag = 0;
				}
			}
			*/
			if (nonacceptcont == 50) {
				printf_s("\n=================\nReverting to bestsol.\n=================\n");
				sol = bestsol;// memcpy(&sol, &bestsol, sizeof(bestsol));
				NewSol = sol;
				bestsolreuseflag = 1;
				nonacceptcont = 0;
			}
			else {
				bestsolreuseflag = 0;
			}
			char TACTIMEHX2[400];
			//bestsol, nhe, bestnhe
			sprintf_s(TACTIMEHX2, "%i\t%.2f\t%.2f\t%i\n", (int)nhe, sol.TotalCosts, bestsol.TotalCosts, (int)bestnhe);
			myfile2 << TACTIMEHX2;

			for (k = 0; k <= nstagesind; k++) {
				for (j = 0; j <= ncoldind; j++) {
					for (i = 0; i <= nhotind; i++) {
						//if (z[i][j][k] == 1) {
						bestsolCont.Q[i][j][k] = 0;
						bestsolCont.Fh[i][j][k] = 0;
						bestsolCont.Fc[i][j][k] = 0;
						GlobalBest.Q[i][j][k] = 0;
						GlobalBest.Fh[i][j][k] = 0;
						GlobalBest.Fc[i][j][k] = 0;

						//}
					}
				}
			}

			for (p = 0; p <= Particles - 1; p++) {
				for (k = 0; k <= nstagesind; k++) {
					for (j = 0; j <= ncoldind; j++) {
						for (i = 0; i <= nhotind; i++) {
							//if (z[i][j][k] == 1) {
							Particle[p].Q[i][j][k] = 0;
							Particle[p].Fh[i][j][k] = 0;
							Particle[p].Fc[i][j][k] = 0;
							Particle[p].VelQ[i][j][k] = 0;
							Particle[p].VelFh[i][j][k] = 0;
							Particle[p].VelFc[i][j][k] = 0;
							ParticleBest[p].Q[i][j][k] = 0;
							ParticleBest[p].Fh[i][j][k] = 0;
							ParticleBest[p].Fc[i][j][k] = 0;
							//}
						}
					}
				}
			}
			/*
			if (ActivateSpaghetti == 1 && finalattempt == 1) {
				finalattempt = 0;
			}
			*/
			end = clock();
			time = (double)(end - start) / CLOCKS_PER_SEC;
			printf("==========================================================\nNewSol = %.2f | Sol = %.2f | Best = %.2f\nNHE = %.0f | SA Temp. Iter. = %i | Attempt no. = %i/%i\nLast attempted match = %i %i %i\nTime = %.3f\n%.6f\n==========================================================", GlobalBest.TotalCosts, sol.TotalCosts, bestsol.TotalCosts, nhe, l + 1, attcont + 1, 0, randomh + 1, randomc + 1, randoms + 1, time);
		}

		p = 0;
		//memcpy(&Particle[0], &bestsol, sizeof(bestsol));
		/*
		somacalor = 0;
		for (j = 0; j <= ncoldind; j++) {
			for (k = 0; k <= nstagesind; k++) {
				for (i = 0; i <= nhotind; i++) {
					if (Particle[0].Q[i][j][k] > Qmin || Particle[0].Q[i][j][k] < -Qmin) {
						somacalor = somacalor + Particle[0].Q[i][j][k];
					};
				};
			};
		};
		if (somacalor < Qmin && attcont == 22 && Particle[0].TotalCosts < 10000000) {
			AreaCosts = AreaCosts;
		}
		*/
		int aaaaaaa = 0;
		//if (aaaaaaa = 1) {
			//Particle[0].Q[0][0][0] = 303.473000000033; Particle[0].Fh[0][0][0] = 303.473000000033 / (606.946000000096 + 303.473000000033); Particle[0].Fc[0][0][0] = 1;
			//Particle[0].Q[0][4][0] = 606.946000000096; Particle[0].Fh[0][4][0] = 606.946000000096 / (606.946000000096 + 303.473000000033); Particle[0].Fc[0][4][0] = 1;
		//}
		//goto TESTOBJFUN;

	//resumeprint:
		SWS(spaghetti, nstages, CaseStudyHEN, bestsol, HSI, 0);

		//if (ActivateSpaghetti == 1 && finalattempt == 1) {
			//sprintf_s(buffer, "Solutions\\SPAGHETTI %ih%ic %i-%i-%i %ih%im%is T%i TAC %i.txt", nhot, ncold, timeinfo.tm_mday, timeinfo.tm_mon + 1, timeinfo.tm_year - 100, timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec, (int)Tk, (int)bestsol.TotalCosts);
		//}
		//else {
		sprintf_s(buffer, "Solutions\\%ih%ic %i-%i-%i %ih%im%is T%i P%i TAC %i.txt", nhot, ncold, timeinfo.tm_mday, timeinfo.tm_mon + 1, timeinfo.tm_year - 100, timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec, (int)Tk, (int)attcont, (int)bestsol.TotalCosts);
		//}
		char QFF[500];
		ofstream myfile(buffer);
		if (myfile.is_open()) {
			sprintf_s(QFF, "Time = \t%.3f\n", bestsoltime);
			myfile << QFF;
			sprintf_s(QFF, "Stages = \t%i\tcT0 = \t%.4f\tcTfinal = %.4f\tcL = \t%.4f\tT0 = \t%.2f\tTf = \t%.4f\tL = \t%.4f\talpha =\t%.4f\tcalpha =\t%.4f\tPSOMaxIter =\t%.4f\tQmin = \t%.6f\tFrmin = \t%.6f\tEMAT = %.2f\t\n", nstages, RFOParam.cT0, RFOParam.cTf, RFOParam.cL, SAParam.T0, Tf, SAParam.L, SAParam.alpha, RFOParam.calpha, RFOParam.PSOMaxIter, Qmin, Frmin, CaseStudyHEN.EMAT);
			myfile << QFF;
			sprintf_s(QFF, "Penalties: %.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t\n", CaseStudyHEN.qaCtroc, 0, 0, CaseStudyHEN.qbCtroc, 0, 0, CaseStudyHEN.taCtroc, CaseStudyHEN.taChu, CaseStudyHEN.taCcu, CaseStudyHEN.tbCtroc, CaseStudyHEN.tbChu, CaseStudyHEN.tbCcu);
			myfile << QFF;
			sprintf_s(QFF, "c1=%.2f\tc2=%.2f\twmin=%.2f\twmax=%.2f\tv0factor=%.2f\tv0ffactor=%.2f\t\n\n", RFOParam.c1, RFOParam.c2, RFOParam.wmin, RFOParam.wmax, RFOParam.v0factor, RFOParam.v0ffactor);
			myfile << QFF;


			sprintf_s(QFF, "i\tj\tk\tQ\tFh\tFc\tThin\tThout\tTcin\tTcout\tLMTD\tArea\n");
			myfile << QFF;

			for (k = 0; k <= nstagesind; k++) {
				for (i = 0; i <= nhotind; i++) {
					for (j = 0; j <= ncoldind; j++) {
						if (bestsol.Q[i][j][k] > Qmin) {
							sprintf_s(QFF, "%i\t%i\t%i\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t\n", i, j, k, bestsol.Q[i][j][k], bestsol.Fh[i][j][k], bestsol.Fc[i][j][k], HSI.Thk[i][k], HSI.Thout[i][j][k], HSI.Tck[j][k], HSI.Tcout[i][j][k], HSI.LMTD[i][j][k], HSI.Area[i][j][k]);
							//printf("Thin = %.4f | Tcin = %.4f \n", Thk[i][k], Tck[i][k]);
							//printf("Thout = %.4f | Tcout = %.4f \n", Thout[i][j][k], Tcout[i][j][k]);
							//printf("DTML = %.4f | Area = %.4f\n", LMTD[i][j][k], Area[i][j][k]);
							myfile << QFF;
						}
					}
				}
			}

			sprintf_s(QFF, "\ni\tQcu\tThin\tThout\tLMTDcu\tAreacu\n");
			myfile << QFF;
			for (i = 0; i <= nhotind; i++) {
				sprintf_s(QFF, "%i\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t\n", i, HSI.Qcu[i], HSI.Thfinal0[i], CaseStudyHEN.Thfinal[i], HSI.LMTDcu[i], HSI.Areacu[i]);
				myfile << QFF;
			}

			sprintf_s(QFF, "\nj\tQhu\tTcin\tTcout\tLMTDhu\tAreahu\n");
			myfile << QFF;
			for (j = 0; j <= ncoldind; j++) {
				sprintf_s(QFF, "%i\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t\n", j, HSI.Qhu[j], HSI.Tcfinal0[j], CaseStudyHEN.Tcfinal[j], HSI.LMTDhu[j], HSI.Areahu[j]);
				myfile << QFF;
			}

			sprintf_s(QFF, "\nTotal Costs = \t%.12f", bestsol.TotalCosts);
			myfile << QFF;
			sprintf_s(QFF, "\nTPenCosts = \t%.12f", bestsol.TPenCosts);
			myfile << QFF;

			sprintf_s(QFF, "\nInput:\n\n");
			myfile << QFF;
			/*
			if (ActivateSpaghetti == 1 && finalattempt == 1) {
				for (k = 0; k <= genk; k++) {
					for (j = 0; j <= ncoldind; j++) {
						for (i = 0; i <= nhotind; i++) {
							if (Qgeneral[i][j][k] > Qmin) {
								sprintf_s(QFF, "i=%i; j=%i; k=%i; Qgeneral[i][j][k]=%.12f; Fhgeneral[i][j][k]=%.12f; Fcgeneral[i][j][k]=%.12f;\n", i, j, k, Qgeneral[i][j][k], Fhgeneral[i][j][k], Fcgeneral[i][j][k]);
								//printf("Thin = %.4f | Tcin = %.4f \n", Thk[i][k], Tck[i][k]);
								//printf("Thout = %.4f | Tcout = %.4f \n", Thout[i][j][k], Tcout[i][j][k]);
								//printf("DTML = %.4f | Area = %.4f\n", LMTD[i][j][k], Area[i][j][k]);
								myfile << QFF;
							}
						}
					}
				}
			}
			else {
				for (k = 0; k <= nstagesind; k++) {
					for (j = 0; j <= ncoldind; j++) {
						for (i = 0; i <= nhotind; i++) {
							if (bestsol.Q[i][j][k] > Qmin) {
								sprintf_s(QFF, "i=%i; j=%i; k=%i; Particle[p].Q[i][j][k]=%.12f; Particle[p].Fh[i][j][k]=%.12f; Particle[p].Fc[i][j][k]=%.12f;\n", i, j, k, bestsol.Q[i][j][k], bestsol.Fh[i][j][k], bestsol.Fc[i][j][k]);
								//printf("Thin = %.4f | Tcin = %.4f \n", Thk[i][k], Tck[i][k]);
								//printf("Thout = %.4f | Tcout = %.4f \n", Thout[i][j][k], Tcout[i][j][k]);
								//printf("DTML = %.4f | Area = %.4f\n", LMTD[i][j][k], Area[i][j][k]);
								myfile << QFF;
							}
						}
					}
				}
			}
			*/
			for (k = 0; k <= nstagesind; k++) {
				for (j = 0; j <= ncoldind; j++) {
					for (i = 0; i <= nhotind; i++) {
						if (bestsol.Q[i][j][k] > Qmin) {
							sprintf_s(QFF, "i=%i; j=%i; k=%i; Particle[p].Q[i][j][k]=%.12f; Particle[p].Fh[i][j][k]=%.12f; Particle[p].Fc[i][j][k]=%.12f;\n", i, j, k, bestsol.Q[i][j][k], bestsol.Fh[i][j][k], bestsol.Fc[i][j][k]);
							//printf("Thin = %.4f | Tcin = %.4f \n", Thk[i][k], Tck[i][k]);
							//printf("Thout = %.4f | Tcout = %.4f \n", Thout[i][j][k], Tcout[i][j][k]);
							//printf("DTML = %.4f | Area = %.4f\n", LMTD[i][j][k], Area[i][j][k]);
							myfile << QFF;
						}
					}
				}
			}
			/*
			sprintf_s(QFF, "\nFor use with WHEN\n");
			myfile << QFF;
			for (i = 0; i <= nhotind; i++) {
				sprintf_s(QFF, "Particle[p].Qcuinter[%i][0][nstagesind] = %.15f; Particle[p].Fhcuinter[%i][0][nstagesind] = 1.0;\n", i, bestsol.Qcu[i], i);
				myfile << QFF;
			}
			for (j = 0; j <= ncoldind; j++) {
				sprintf_s(QFF, "Particle[p].Qhuinter[0][%i][0] = %.15f; Particle[p].Fchuinter[0][%i][0] = 1.0;\n", j, bestsol.Qhu[j], j);
				myfile << QFF;
			}
			*/
			myfile.close();
			//if (fromspaghetti == 1) {
			//	fromspaghetti = 0;
			//	goto finalattemptinput;
			//}
		}
		else cout << "Unable to open file";

		Tk = SAParam.alpha * Tk;

		HENSolution = bestsol;

		if (spaghetti == 1 && Tk < Tf) {
			double QCUTOT = 0;
			for (int i = 0; i <= bestsol.HSI.Qcu.size() - 1; i++) {
				if (isnan(bestsol.HSI.Qcu[i]) == 0) {
					QCUTOT = QCUTOT + bestsol.HSI.Qcu[i];
				}
			}
			double QHUTOT = 0;
			for (int i = 0; i <= bestsol.HSI.Qhu.size() - 1; i++) {
				if (isnan(bestsol.HSI.Qhu[i]) == 0) {
					QHUTOT = QHUTOT + bestsol.HSI.Qhu[i];
				}
			}
			if (QCUTOT > 0.0001 && QHUTOT > 0.0001) {
				Tk = SAParam.T0;
				bestsol = InitialSol;
				sol = InitialSol;
				NewSol = InitialSol;
				solContVars = InitialSol;
				bestsolCont = InitialSol;
				firstsol = 1;
			}
		}


		if (spaghetti == 0) {
			//SWS(spaghetti, nstages, CaseStudyHEN, sol, HSI, 1);
			SWS(spaghetti, nstages, CaseStudyHEN, bestsol, HSI, 1);
		}
		if (endopt == 1) {
			break;
		}
	}
	SWS(spaghetti, nstages, CaseStudyHEN, bestsol, HSI, 1);


}



// ----- funções carlos ----- //

string obter_path_modelo(string nome_modelo) {
	// obtém o path atual, bem como a posição da pasta TCC neste path
	string path_atual = filesystem::current_path().string();
	size_t pos = path_atual.find("TCC_bruto");

	// corta o path até a parte TCC/
	string path_tcc;
	if (pos != string::npos) {
		path_tcc = path_atual.substr(0, pos + 10);
	}

	// adiciona o arquivo modelos/nome_modelo no path TCC/
	string path_modelo = path_tcc + "modelos\\" + nome_modelo;

	return path_modelo;
}

float inferencia(float input,
	torch::jit::script::Module modelo,
	float normalizacao = 0)
{
	// transforma o tensor do input em tensor, e em seguida em vetor para que o modelo possa utiliza-lo
	torch::Tensor input_tensor = torch::tensor({ {input} });
	vector <torch::jit::IValue> input_vector;
	input_vector.push_back(input_tensor);

	// faz a inferência e normaliza o output (entalpias foram transformadas em positivas pra não dar problema com relu)
	auto output = modelo.forward(input_vector);
	float resultado = output.toTensor().item<float>() + normalizacao;

	return resultado;
}





int main() {
	cout << "Testes Carlos:" << endl;
	string path_modelo = obter_path_modelo("teste.pt");
	torch::jit::script::Module qual_a_enth = torch::jit::load(path_modelo);

	float normalizacao_enth = -89587.13674;

	float teste = inferencia(142,
		qual_a_enth,
		normalizacao_enth);

	cout << teste << endl;



	// ===========================================================================================================================
// 1 - DECLARA��ES PARA OBTER TEMPO DE EXECU��O
// ===========================================================================================================================

	int aaa;
	time_t rawtime;
	struct tm timeinfo;
	time(&rawtime);
	localtime_s(&timeinfo, &rawtime);

	vector<int> today; today.resize(6);
	today[0] = timeinfo.tm_mday;
	today[1] = timeinfo.tm_mon + 1;
	today[2] = timeinfo.tm_year - 100;
	today[3] = timeinfo.tm_hour;
	today[4] = timeinfo.tm_min;
	today[5] = timeinfo.tm_sec;

	clock_t start = clock();
	clock_t end = clock();
	double TotalTime;

	srand(time(NULL));


	int spaghetti = 0;

	vector<CaseStudyHENStruct> CaseStudyHEN;


	HENSynInterm HSIzero;
	HENSynInterm HSI;
	HENSolutionStruct HENSolution;

	int csno = 0;

	//========== BANCO DE DADOS DE CASOS DE ESTUDO ===========
	//=== CASO DE ESTUDO 0 ===================================
	csno = 0;
	CaseStudyHEN.push_back(CaseStudyHENStruct());

	CaseStudyHEN[csno].AF = 1.0;
	CaseStudyHEN[csno].Streams = {
	{ 1, 327, 40, 100, 0.5 },
	{ 2, 220, 160, 160, 0.4 },
	{ 3, 220, 60, 60,  0.14 },
	{ 4, 160, 45, 400, 0.3 },
	{ 5, 100, 300, 100, 0.35 },
	{ 6, 35,  164, 70,  0.70 },
	{ 7, 85,  138, 350, 0.50 },
	{ 8, 60,  170, 60,  0.14 },
	{ 9, 140, 300, 200, 0.60 }
	};

	CaseStudyHEN[csno].HUStreams = {
		{ 1, 330, 250, 0, 0.5 },
	};
	CaseStudyHEN[csno].CUStreams = {
		{ 1, 15, 30, 0, 0.5 },
	};


	CaseStudyHEN[csno].specialHE = 0;

	//Custos de utilidades
	CaseStudyHEN[csno].HUCosts = { 60.0 };
	CaseStudyHEN[csno].CUCosts = { 6.0 };

	//Fatores CC
	//- Correntes de processo
	CaseStudyHEN[csno].B = { 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0 };
	CaseStudyHEN[csno].C = { 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0 };
	CaseStudyHEN[csno].beta = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

	//- UQ
	CaseStudyHEN[csno].Bh = { 2000.0 };
	CaseStudyHEN[csno].Ch = { 70.0 };
	CaseStudyHEN[csno].betah = { 1.0 };

	//- UF
	CaseStudyHEN[csno].Bc = { 2000.0 };
	CaseStudyHEN[csno].Cc = { 70.0 };
	CaseStudyHEN[csno].betac = { 1.0 };

	//- Padr�o (p/ uso no Pinch)
	CaseStudyHEN[csno].B0 = { 2000.0 };
	CaseStudyHEN[csno].C0 = { 70.0 };
	CaseStudyHEN[csno].beta0 = { 1.0 };



	//=============================================================
	//=== CASO DE ESTUDO 1 ========================================

	csno = 1;
	CaseStudyHEN.push_back(CaseStudyHENStruct());

	CaseStudyHEN[csno].AF = 1.0;
	CaseStudyHEN[csno].Streams = {
	{ 1, 327, 40, 100, 0.5 },
	{ 2, 220, 160, 160, 0.4 },
	{ 3, 220, 60, 60,  0.14 },
	{ 4, 160, 45, 400, 0.3 },
	{ 5, 100, 300, 100, 0.35 },
	{ 6, 35,  164, 70,  0.70 },
	{ 7, 85,  138, 350, 0.50 },
	{ 8, 60,  170, 60,  0.14 },
	{ 9, 140, 300, 200, 0.60 }
	};

	CaseStudyHEN[csno].HUStreams = {
		{ 1, 330, 250, 0, 0.5 },
		{ 2, 300, 299, 0, 0.5 },
		{ 3, 200, 199, 0, 0.5 },
	};
	CaseStudyHEN[csno].CUStreams = {
		{ 1, 15, 30, 0, 0.5 },
		{ 2, -5, 0, 0, 0.5 },
	};


	CaseStudyHEN[csno].specialHE = 0;

	//Custos de utilidades
	CaseStudyHEN[csno].HUCosts = { 60.0, 40.0, 30.0 };
	CaseStudyHEN[csno].CUCosts = { 6.0, 20 };

	//Fatores CC
	//- Correntes de processo
	CaseStudyHEN[csno].B = { 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0 };
	CaseStudyHEN[csno].C = { 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0 };
	CaseStudyHEN[csno].beta = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

	//- UQ
	CaseStudyHEN[csno].Bh = { 2000.0, 2000.0, 2000.0 };
	CaseStudyHEN[csno].Ch = { 70.0, 70.0, 70.0 };
	CaseStudyHEN[csno].betah = { 1.0, 1.0, 1.0 };

	//- UF
	CaseStudyHEN[csno].Bc = { 2000.0, 2000.0 };
	CaseStudyHEN[csno].Cc = { 70.0, 70.0 };
	CaseStudyHEN[csno].betac = { 1.0, 1.0 };

	//- Padr�o (p/ uso no Pinch)
	CaseStudyHEN[csno].B0 = { 2000.0 };
	CaseStudyHEN[csno].C0 = { 70.0 };
	CaseStudyHEN[csno].beta0 = { 1.0 };


	//=============================================================
	//=== CASO DE ESTUDO 2 ========================================
	//...
	//=============================================================
	//=============================================================
	//=== CASO DE ESTUDO 3 ========================================
	//...
	//=============================================================

	csno = 1; //declarar caso a ser usado

	//PROCESSAMENTO DO CASO DE ESTUDO
	//Esta fun��o organiza os dados para deixar no formato certo para s�ntese de RTC (Sem An�lise Pinch)
	BuildHENSyn0(5, CaseStudyHEN[csno], HSIzero, HENSolution);
	HENSynInterm HSIspag = HSIzero; HENSolutionStruct HENSolutionSpag;
	BuildHENSyn0(1, CaseStudyHEN[csno], HSIspag, HENSolutionSpag);

	//==================================================================================================
	// 1 - Otimiza��o do dTmin e das fra��es de utilidades
	//==================================================================================================

	CaseStudyStruct OutCaseStudy;
	SolutionStruct BestSol;
	CaseStudyPinch CSPinch;
	CaseStudyPinch CSPinchzero;
	PinchInterm PI; PinchInterm PIzero;
	PinchSolution TrivialPS; PinchSolution TrivialPSzero;

	BuildPinchCS(BestSol, CaseStudyHEN[csno], CSPinch, CSPinchzero);

	ResizePinch(CSPinch, BestSol.PS, PIzero); //ResizePinch s� ser� utilizada ap�s mudan�a de estrutura
	PI = PIzero;
	BestSol.PS.dTmin = 24;
	BestSol.PS.HUFrac[0] = 1.0;
	BestSol.PS.CUFrac[0] = 1.0;

	Pinch(CSPinch, BestSol.PS, PI, 1, start, today);
	BestSol.PS.TotalOC = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts);
	BestSol.PS.TotalCC = CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
	BestSol.TotalCosts = BestSol.PS.TotalOC + BestSol.PS.TotalCC;
	SolutionStruct NewSolution = BestSol;

	double dTminub = 45;
	double dTminlb = 1;
	double fraclb = 0.0;
	double fracub = 1.0;

	CSA(10000, 1000, 0.7, 100, 1, BestSol, BestSol, PIzero, OutCaseStudy, CSPinch, CSPinchzero, NewSolution, dTminub, dTminlb, fraclb, fracub, start, today);
	PSO(50, 0.005, 0.25, 0.75, 1.1, 1.1, 500, 1, NewSolution, NewSolution, PIzero, OutCaseStudy, CSPinch, CSPinchzero, BestSol, dTminub, dTminlb, fraclb, fracub, start, today);

	Pinch(CSPinch, BestSol.PS, PI, 1, start, today);
	BestSol.PS.TotalOC = OCHI(PI.HU, PI.CU, CSPinch.HUcosts, CSPinch.CUcosts);
	BestSol.PS.TotalCC = CCHI(PI.Nun, PI.intervals, PI.Areak, CSPinch.B0, CSPinch.C0, CSPinch.beta0);
	BestSol.TotalCosts = BestSol.PS.TotalOC + BestSol.PS.TotalCC;

	//==================================================================================================
	// 2 - Forma��o da rede "spaghetti"
	//==================================================================================================


	//	BuildHENSyn(BestSol.PS, 5, CaseStudyHEN[csno], HSIzero, HENSolution, CSPinch, PI);

	CaseStudyHEN[csno].taCtroc = 10000000;
	CaseStudyHEN[csno].tbCtroc = 10000000;

	CaseStudyHEN[csno].qaCtroc = 10000000;
	CaseStudyHEN[csno].qbCtroc = 10000000;

	CaseStudyHEN[csno].taCcu = 10000000;
	CaseStudyHEN[csno].tbCcu = 10000000;

	CaseStudyHEN[csno].taChu = 10000000;
	CaseStudyHEN[csno].tbChu = 10000000;

	SAParamStruct SAParam;
	SAParam.T0 = 10000;
	SAParam.Tf = 100;
	SAParam.L = 50;
	SAParam.alpha = 0.5;
	SAParam.fixtopology = 0;
	SAParam.fixedz = HENSolution.z;
	SAParam.Forbidh.resize(CaseStudyHEN[csno].AllHotStreams.size());
	SAParam.Forbidc.resize(CaseStudyHEN[csno].AllColdStreams.size());
	SAParam.maxHE = 100;

	RFOParamStruct RFOParam;
	RFOParam.cT0 = 10000;
	RFOParam.cTf = 7000;
	RFOParam.cL = 500;
	RFOParam.slowingfactor = 0.1;
	RFOParam.calpha = 0.9;

	RFOParam.Particles = 50;
	RFOParam.PSOMaxIter = 100;
	RFOParam.c1 = 1.1;
	RFOParam.c2 = 1.1;
	RFOParam.wmin = 0.5;
	RFOParam.wmax = 0.75;
	RFOParam.v0factor = 0.1;
	RFOParam.v0ffactor = 0.1;

	int TrivialOnly = 1;
	int SpaghettiOnly = 0;
	int SemiSpaghettiOnly = 0;

	//==================================================================================================
	// 3 - Otimiza��o a partir da Rede trivial
	//==================================================================================================

	if (TrivialOnly == 1) {
		HENSynInterm HSIzero2 = HSIzero;
		HENSolutionStruct HENSolution2 = HENSolution;

		sprintf_s(CaseStudyHEN[csno].comment, "Triv");
		SARFO(0, 0, 5, SAParam, RFOParam, CaseStudyHEN[csno], HSIzero2, HENSolution2, today, start);
	}


	//==================================================================================================
	// 4 - Otimiza��o a partir da Rede "spaghetti"
	//==================================================================================================

	if (SpaghettiOnly == 1) {
		vector<vector<vector<double>>> Streams2;

		BuildSpagCases(CSPinch, PI, CaseStudyHEN[csno], Streams2);

		double sumheath = 0;
		double sumheatc = 0;
		int attcont = 0;
		int TotalAreaIntervals = PI.intervals + 2;

		CaseStudyHENStruct TempCaseStudyHEN;
		TempCaseStudyHEN = CaseStudyHEN[csno];

		HENSolutionStruct SpaghettiSolution;
		SpaghettiSolution = HENSolution;

		HENSolutionStruct HENSolutionzero;
		HENSolutionzero = HENSolution;



		vector<int> Forbidh; Forbidh.resize(CaseStudyHEN[csno].nhot + CaseStudyHEN[csno].nhu);
		vector<int> Forbidc; Forbidc.resize(CaseStudyHEN[csno].ncold + CaseStudyHEN[csno].ncu);
		int curstage = 0;
		while (attcont <= TotalAreaIntervals - 2) {
			while ((sumheath == 0 && sumheatc == 0) && attcont <= TotalAreaIntervals - 2) {
				int conth = 0;
				for (conth = 0; conth <= CaseStudyHEN[csno].nhot + CaseStudyHEN[csno].nhu - 1; conth++) {
					if (Streams2[attcont][conth][3] * (Streams2[attcont][conth][1] - Streams2[attcont][conth][2]) < CaseStudyHEN[csno].Qmin) {
						Forbidh[conth] = 1;
					}
					else {
						Forbidh[conth] = 0;
						sumheath = sumheath + Streams2[attcont][conth][3] * (Streams2[attcont][conth][1] - Streams2[attcont][conth][2]);
					}
				}
				for (int contc = 0; contc <= CaseStudyHEN[csno].ncold + CaseStudyHEN[csno].ncu - 1; contc++) {
					if (Streams2[attcont][contc + conth][3] * (Streams2[attcont][contc + conth][2] - Streams2[attcont][contc + conth][1]) < CaseStudyHEN[csno].Qmin) {
						Forbidc[contc] = 1;
					}
					else {
						Forbidc[contc] = 0;
						sumheatc = sumheatc + Streams2[attcont][contc + conth][3] * (Streams2[attcont][contc + conth][2] - Streams2[attcont][contc + conth][1]);
					}

				}

				if (sumheath > 0 && sumheatc > 0) {
					HENSolution = HENSolutionzero;
					int cont = 0;
					for (cont = 0; cont <= CaseStudyHEN[csno].nhot + CaseStudyHEN[csno].nhu - 1; cont++) {
						TempCaseStudyHEN.AllHotStreams[cont][0] = Streams2[attcont][cont][0];
						TempCaseStudyHEN.AllHotStreams[cont][1] = Streams2[attcont][cont][1];
						TempCaseStudyHEN.AllHotStreams[cont][2] = Streams2[attcont][cont][2];
						TempCaseStudyHEN.AllHotStreams[cont][3] = Streams2[attcont][cont][3];
						TempCaseStudyHEN.AllHotStreams[cont][4] = Streams2[attcont][cont][4];
						TempCaseStudyHEN.Qh[cont] = TempCaseStudyHEN.AllHotStreams[cont][3] * (TempCaseStudyHEN.AllHotStreams[cont][1] - TempCaseStudyHEN.AllHotStreams[cont][2]);
						TempCaseStudyHEN.Qhk[cont] = TempCaseStudyHEN.Qh[cont];
						TempCaseStudyHEN.CPh[cont] = TempCaseStudyHEN.AllHotStreams[cont][3];
						TempCaseStudyHEN.Thin[cont] = TempCaseStudyHEN.AllHotStreams[cont][1];
						TempCaseStudyHEN.Thfinal[cont] = TempCaseStudyHEN.AllHotStreams[cont][2];
					}
					int jj = 0;
					for (cont = cont; cont <= CaseStudyHEN[csno].nhot + CaseStudyHEN[csno].nhu + CaseStudyHEN[csno].ncold + CaseStudyHEN[csno].ncu - 1; cont++) {
						TempCaseStudyHEN.AllColdStreams[jj][0] = Streams2[attcont][cont][0];
						TempCaseStudyHEN.AllColdStreams[jj][1] = Streams2[attcont][cont][1];
						TempCaseStudyHEN.AllColdStreams[jj][2] = Streams2[attcont][cont][2];
						TempCaseStudyHEN.AllColdStreams[jj][3] = Streams2[attcont][cont][3];
						TempCaseStudyHEN.AllColdStreams[jj][4] = Streams2[attcont][cont][4];
						TempCaseStudyHEN.Qc[jj] = TempCaseStudyHEN.AllColdStreams[jj][3] * (TempCaseStudyHEN.AllColdStreams[jj][2] - TempCaseStudyHEN.AllColdStreams[jj][1]);
						TempCaseStudyHEN.Qck[jj] = TempCaseStudyHEN.Qc[jj];
						TempCaseStudyHEN.CPc[jj] = TempCaseStudyHEN.AllColdStreams[jj][3];
						TempCaseStudyHEN.Tcin[jj] = TempCaseStudyHEN.AllColdStreams[jj][1];
						TempCaseStudyHEN.Tcfinal[jj] = TempCaseStudyHEN.AllColdStreams[jj][2];

						jj++;
					}
				}
				else {
					attcont++;
				}

			}

			if (attcont <= TotalAreaIntervals - 2) {
				sumheath = 0; sumheatc = 0;

				SAParamStruct SpagSAParam;
				SpagSAParam.T0 = 10000;
				SpagSAParam.Tf = 1000;
				SpagSAParam.L = 10;
				SpagSAParam.alpha = 0.9;
				SpagSAParam.fixtopology = 0;
				SpagSAParam.fixedz = HENSolution.z;
				SpagSAParam.Forbidh = Forbidh;
				SpagSAParam.Forbidc = Forbidc;
				SpagSAParam.maxHE = 100;

				RFOParamStruct SpagRFOParam;
				SpagRFOParam.cT0 = 10000;
				SpagRFOParam.cTf = 7000;
				SpagRFOParam.cL = 50;
				SpagRFOParam.slowingfactor = 0.1;
				SpagRFOParam.calpha = 0.9;

				SpagRFOParam.Particles = 50;
				SpagRFOParam.PSOMaxIter = 100;
				SpagRFOParam.c1 = 1.1;
				SpagRFOParam.c2 = 1.1;
				SpagRFOParam.wmin = 0.5;
				SpagRFOParam.wmax = 0.75;
				SpagRFOParam.v0factor = 0.1;
				SpagRFOParam.v0ffactor = 0.1;


				HSI = HSIzero;

				sprintf_s(TempCaseStudyHEN.comment, "Sg%i", curstage);
				SARFO(0, 1, 1, SpagSAParam, SpagRFOParam, TempCaseStudyHEN, HSI, HENSolution, today, start);
				if (curstage == 0) {
					SpaghettiSolution = HENSolution;
				}
				else {
					for (int i = 0; i <= HENSolution.Q.size() - 1; i++) {
						for (int j = 0; j <= HENSolution.Q[i].size() - 1; j++) {
							SpaghettiSolution.Fc[i][j].push_back(HENSolution.Fc[i][j][0]);
							SpaghettiSolution.Fh[i][j].push_back(HENSolution.Fh[i][j][0]);
							SpaghettiSolution.Q[i][j].push_back(HENSolution.Q[i][j][0]);
							SpaghettiSolution.z[i][j].push_back(HENSolution.z[i][j][0]);
						}
					}

					SpaghettiSolution.VelFc = SpaghettiSolution.Fc;
					SpaghettiSolution.VelFh = SpaghettiSolution.Fh;
					SpaghettiSolution.VelQ = SpaghettiSolution.Q;
				}

				attcont++;
				curstage++;
			}
		}

		HENSynInterm HSIspag;
		HENSolutionStruct Solspag;
		CaseStudyHENStruct CaseStudyHENSpag;
		HENSynInterm HSIspagzero;

		int spagstag = SpaghettiSolution.Q[0][0].size();
		BuildHENSyn0(spagstag, CaseStudyHEN[csno], HSIspag, SpaghettiSolution);
		sprintf_s(CaseStudyHEN[csno].comment, "Spg0");
		SWS(0, spagstag, CaseStudyHEN[csno], SpaghettiSolution, HSIspag, 1);

		sprintf_s(CaseStudyHEN[csno].comment, "Spg");

		SARFO(1, 0, spagstag, SAParam, RFOParam, CaseStudyHEN[csno], HSIspag, SpaghettiSolution, today, start);
	}

	//==================================================================================================
	// 5 - Otimiza��o a partir da Rede "semi-spaghetti"
	//==================================================================================================

	if (SemiSpaghettiOnly == 1) {
		vector<vector<vector<double>>> Streams2;

		BuildSpagCases(CSPinch, PI, CaseStudyHEN[csno], Streams2);

		double sumheath = 0;
		double sumheatc = 0;
		int attcont = 0;
		int TotalAreaIntervals = PI.intervals + 2;

		CaseStudyHENStruct TempCaseStudyHEN;
		TempCaseStudyHEN = CaseStudyHEN[csno];

		HENSolutionStruct SpaghettiSolution;
		SpaghettiSolution = HENSolutionSpag;

		HENSolutionStruct HENSolutionzero;
		HENSolutionzero = HENSolutionSpag;



		vector<int> Forbidh; Forbidh.resize(CaseStudyHEN[csno].nhot + CaseStudyHEN[csno].nhu);
		vector<int> Forbidc; Forbidc.resize(CaseStudyHEN[csno].ncold + CaseStudyHEN[csno].ncu);
		int curstage = 0;
		while (attcont <= TotalAreaIntervals - 2) {
			while ((sumheath == 0 && sumheatc == 0) && attcont <= TotalAreaIntervals - 2) {
				int conth = 0;
				for (conth = 0; conth <= CaseStudyHEN[csno].nhot + CaseStudyHEN[csno].nhu - 1; conth++) {
					if (Streams2[attcont][conth][3] * (Streams2[attcont][conth][1] - Streams2[attcont][conth][2]) < CaseStudyHEN[csno].Qmin) {
						Forbidh[conth] = 1;
					}
					else {
						Forbidh[conth] = 0;
						sumheath = sumheath + Streams2[attcont][conth][3] * (Streams2[attcont][conth][1] - Streams2[attcont][conth][2]);
					}
				}
				for (int contc = 0; contc <= CaseStudyHEN[csno].ncold + CaseStudyHEN[csno].ncu - 1; contc++) {
					if (Streams2[attcont][contc + conth][3] * (Streams2[attcont][contc + conth][2] - Streams2[attcont][contc + conth][1]) < CaseStudyHEN[csno].Qmin) {
						Forbidc[contc] = 1;
					}
					else {
						Forbidc[contc] = 0;
						sumheatc = sumheatc + Streams2[attcont][contc + conth][3] * (Streams2[attcont][contc + conth][2] - Streams2[attcont][contc + conth][1]);
					}

				}

				if (sumheath > 0 && sumheatc > 0) {
					HENSolution = HENSolutionzero;
					int cont = 0;
					for (cont = 0; cont <= CaseStudyHEN[csno].nhot + CaseStudyHEN[csno].nhu - 1; cont++) {
						TempCaseStudyHEN.AllHotStreams[cont][0] = Streams2[attcont][cont][0];
						TempCaseStudyHEN.AllHotStreams[cont][1] = Streams2[attcont][cont][1];
						TempCaseStudyHEN.AllHotStreams[cont][2] = Streams2[attcont][cont][2];
						TempCaseStudyHEN.AllHotStreams[cont][3] = Streams2[attcont][cont][3];
						TempCaseStudyHEN.AllHotStreams[cont][4] = Streams2[attcont][cont][4];
						TempCaseStudyHEN.Qh[cont] = TempCaseStudyHEN.AllHotStreams[cont][3] * (TempCaseStudyHEN.AllHotStreams[cont][1] - TempCaseStudyHEN.AllHotStreams[cont][2]);
						TempCaseStudyHEN.Qhk[cont] = TempCaseStudyHEN.Qh[cont];
						TempCaseStudyHEN.CPh[cont] = TempCaseStudyHEN.AllHotStreams[cont][3];
						TempCaseStudyHEN.Thin[cont] = TempCaseStudyHEN.AllHotStreams[cont][1];
						TempCaseStudyHEN.Thfinal[cont] = TempCaseStudyHEN.AllHotStreams[cont][2];
					}
					int jj = 0;
					for (cont = cont; cont <= CaseStudyHEN[csno].nhot + CaseStudyHEN[csno].nhu + CaseStudyHEN[csno].ncold + CaseStudyHEN[csno].ncu - 1; cont++) {
						TempCaseStudyHEN.AllColdStreams[jj][0] = Streams2[attcont][cont][0];
						TempCaseStudyHEN.AllColdStreams[jj][1] = Streams2[attcont][cont][1];
						TempCaseStudyHEN.AllColdStreams[jj][2] = Streams2[attcont][cont][2];
						TempCaseStudyHEN.AllColdStreams[jj][3] = Streams2[attcont][cont][3];
						TempCaseStudyHEN.AllColdStreams[jj][4] = Streams2[attcont][cont][4];
						TempCaseStudyHEN.Qc[jj] = TempCaseStudyHEN.AllColdStreams[jj][3] * (TempCaseStudyHEN.AllColdStreams[jj][2] - TempCaseStudyHEN.AllColdStreams[jj][1]);
						TempCaseStudyHEN.Qck[jj] = TempCaseStudyHEN.Qc[jj];
						TempCaseStudyHEN.CPc[jj] = TempCaseStudyHEN.AllColdStreams[jj][3];
						TempCaseStudyHEN.Tcin[jj] = TempCaseStudyHEN.AllColdStreams[jj][1];
						TempCaseStudyHEN.Tcfinal[jj] = TempCaseStudyHEN.AllColdStreams[jj][2];

						jj++;
					}
				}
				else {
					attcont++;
				}

			}

			if (attcont <= TotalAreaIntervals - 2) {
				sumheath = 0; sumheatc = 0;

				SAParamStruct SpagSAParam;
				SpagSAParam.T0 = 10000;
				SpagSAParam.Tf = 1000;
				SpagSAParam.L = 10;
				SpagSAParam.alpha = 0.9;
				SpagSAParam.fixtopology = 0;
				SpagSAParam.fixedz = HENSolution.z;
				SpagSAParam.Forbidh = Forbidh;
				SpagSAParam.Forbidc = Forbidc;
				SpagSAParam.maxHE = 100;

				RFOParamStruct SpagRFOParam;
				SpagRFOParam.cT0 = 10000;
				SpagRFOParam.cTf = 7000;
				SpagRFOParam.cL = 50;
				SpagRFOParam.slowingfactor = 0.1;
				SpagRFOParam.calpha = 0.9;

				SpagRFOParam.Particles = 50;
				SpagRFOParam.PSOMaxIter = 100;
				SpagRFOParam.c1 = 1.1;
				SpagRFOParam.c2 = 1.1;
				SpagRFOParam.wmin = 0.5;
				SpagRFOParam.wmax = 0.75;
				SpagRFOParam.v0factor = 0.1;
				SpagRFOParam.v0ffactor = 0.1;


				HSI = HSIzero;

				sprintf_s(TempCaseStudyHEN.comment, "SSg%i", curstage);

				int UtilStage = 0;
				int cont = 0;
				for (cont = 0; cont <= CaseStudyHEN[csno].nhot + CaseStudyHEN[csno].nhu - 1; cont++) {
					if (TempCaseStudyHEN.Qh[cont] > 0.0 && TempCaseStudyHEN.ishotutil[cont] > 0) {
						UtilStage = 1;
					}
				}
				int jj = 0;
				for (cont = cont; cont <= CaseStudyHEN[csno].nhot + CaseStudyHEN[csno].nhu + CaseStudyHEN[csno].ncold + CaseStudyHEN[csno].ncu - 1; cont++) {
					if (TempCaseStudyHEN.Qc[jj] > 0.0 && TempCaseStudyHEN.iscoldutil[jj] > 0) {
						UtilStage = 1;
					}
					jj++;
				}
				if (UtilStage == 0) {
					if (curstage == 0) {
						for (int i = 0; i <= HENSolution.Q.size() - 1; i++) {
							for (int j = 0; j <= HENSolution.Q[i].size() - 1; j++) {
								SpaghettiSolution.Fc[i][j].push_back(0.0);
								SpaghettiSolution.Fh[i][j].push_back(0.0);
								SpaghettiSolution.Q[i][j].push_back(0.0);
								SpaghettiSolution.z[i][j].push_back(0);
							}
						}

						SpaghettiSolution.VelFc = SpaghettiSolution.Fc;
						SpaghettiSolution.VelFh = SpaghettiSolution.Fh;
						SpaghettiSolution.VelQ = SpaghettiSolution.Q;
					}
				}
				if (UtilStage == 1) {
					SARFO(0, 1, 1, SpagSAParam, SpagRFOParam, TempCaseStudyHEN, HSI, HENSolutionSpag, today, start);

					if (curstage == 0) {
						SpaghettiSolution = HENSolutionSpag;
					}
					else {
						for (int i = 0; i <= HENSolutionSpag.Q.size() - 1; i++) {
							for (int j = 0; j <= HENSolutionSpag.Q[i].size() - 1; j++) {
								SpaghettiSolution.Fc[i][j].push_back(HENSolutionSpag.Fc[i][j][0]);
								SpaghettiSolution.Fh[i][j].push_back(HENSolutionSpag.Fh[i][j][0]);
								SpaghettiSolution.Q[i][j].push_back(HENSolutionSpag.Q[i][j][0]);
								SpaghettiSolution.z[i][j].push_back(HENSolutionSpag.z[i][j][0]);
							}
						}

						SpaghettiSolution.VelFc = SpaghettiSolution.Fc;
						SpaghettiSolution.VelFh = SpaghettiSolution.Fh;
						SpaghettiSolution.VelQ = SpaghettiSolution.Q;

						for (int i = 0; i <= HENSolution.Q.size() - 1; i++) {
							for (int j = 0; j <= HENSolution.Q[i].size() - 1; j++) {
								SpaghettiSolution.Fc[i][j].push_back(0.0);
								SpaghettiSolution.Fh[i][j].push_back(0.0);
								SpaghettiSolution.Q[i][j].push_back(0.0);
								SpaghettiSolution.z[i][j].push_back(0);
							}
						}

						SpaghettiSolution.VelFc = SpaghettiSolution.Fc;
						SpaghettiSolution.VelFh = SpaghettiSolution.Fh;
						SpaghettiSolution.VelQ = SpaghettiSolution.Q;

					}

				}

				attcont++;
				curstage++;
			}
		}

		HENSynInterm HSIspag;
		HENSolutionStruct Solspag;
		CaseStudyHENStruct CaseStudyHENSpag;
		HENSynInterm HSIspagzero;

		int spagstag = SpaghettiSolution.Q[0][0].size();
		BuildHENSyn0(spagstag, CaseStudyHEN[csno], HSIspag, SpaghettiSolution);
		sprintf_s(CaseStudyHEN[csno].comment, "Spg0");
		SWS(0, spagstag, CaseStudyHEN[csno], SpaghettiSolution, HSIspag, 1);

		sprintf_s(CaseStudyHEN[csno].comment, "Spg");

		SARFO(1, 0, spagstag, SAParam, RFOParam, CaseStudyHEN[csno], HSIspag, SpaghettiSolution, today, start);
	}


	return 0;
}
