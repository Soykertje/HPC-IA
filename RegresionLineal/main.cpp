/************************************************************
 * Fecha: 02-03-2022
 * Autor: Carlos Bermudez
 * Materia: HPC-2 Metricas de rendimiento
 * Tema: Introduccion a Machine Learning.
 * Objetivo: Función principal para el cálculo del modelo de
 * regresion lineal
 * Requerimientos:
 * 1. Aplicacion o clase para la lectura de ficheros CSV
 * presente en una clase para la extraccion de datos, la normalizacion
 * de datos y en general para la manipulacion de datos.
 * 2. Crear una clase para el calculo de la regresion lineal
 *****************************************************************/

#include "extraerdata/extraerdata.h"
#include "linearregresion.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>



int main(int argc, char *argv[])
{
    // se crea un objeto del tipo extraerdata/
    ExtraerData extraer(argv[1],argv[2],argv[3]);
    // Se crea el  objeto de tipo Regresion lineal
    LinealRegresion LR;

    // primer argumento el nombre del dataset
    // Luego el delimitador
    // ../DataSet/winedata.csv ","false
    std::cout << argv[1] << std::endl;
    std::cout << argv[2] << std::endl;
    std::cout << argv[3] << std::endl;

    // leer los datos del fichero, por la funcion ReadCSV() del objetcto extraer

    std::vector<std::vector<std::string>> DataFrame = extraer.ReadCSV();
    /*Para probar la funcion Eifentofile, y de esta manera imprimir el fichero
    de datos, se debe definir el numero de filas y del columnas del dataset. Basado
    en los argumentos de entrada --*/

    int filas = DataFrame.size() +  1;
    int columnas = DataFrame[0].size();
    std::cout << "filas : "<< filas << std::endl;
    std::cout << "col : "<< columnas << std::endl;
    Eigen::MatrixXd MatDataFrame = extraer.CSVtoEigen(DataFrame,filas,columnas);

    // se imprime el objeto Matriz dataframe

    //std::cout << MatDataFrame << std::endl;

    /* Se imprime el vector de promedios por columna */
    // std::cout << extraer.Promedio(MatDataFrame) << std::endl;
    // std::cout << extraer.Promedio(MatDataFrame) << std::endl;

    /* Se crea una matrix para almacnar la data normalizada */
    Eigen::MatrixXd DataNormalizado = extraer.Normalizador(MatDataFrame);
    /*-- Se imprime los Datos Normalizados --*/
    //std::cout << DataNormalizado << std::endl;

    /* A continuación se dividen en grupos de entrenamiento y prueba la
     * matriz dataNorm. Se tomará para entrenamiento el 80% de los datos.
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
            divDatos = extraer.TrainTestSplit(DataNormalizado,0.8);

    Eigen::MatrixXd X_train;
    Eigen::MatrixXd y_train;
    Eigen::MatrixXd X_test;
    Eigen::MatrixXd y_test;

    std::tie(X_train,y_train,X_test,y_test)=divDatos;

    /* Se imprime el numero de filas de cada uno de los elementos. */
    /*std::cout <<std::endl << std::endl;
    std::cout << DataNormalizado.rows() << std::endl;
    std::cout<<X_train.rows()<<std::endl;*/

    /* A continuación se desarrolla el primer algoritmo de Machine Learning
     * REGRESION LINEAL.
     * para el algoritmo s eusa como ejemplo el dataset del VinoRojo, el cual
     * tiene múltiples variables. Dada la naturaleza de RL, si se tiene variables
     * con diferentres unidades (órdenes de magnitud), una variable podría beneficiar
     * /perjudicar a otra variable: Para ello se recomienda estandarizar los datos, dando
     * a todas las variables el mismo orden de magnitud y centradas en 0.
     * Es importante recalcar que la clase artesanal para la manipulación/tratamiento de datos
     * debe ser observada de cerca, a la hora de utilizar cualquier otro dataset.
     */

    /* Se implementa el módulo de ML como clase de RL con su correspondiente interfaz, se
     * define constructor y todas las funciones o métodos necesarios para la RL. Se tiene
     * en cuenta que la RL es un método que define la relación entre las
     * variables independientes y la variable dependiente. La idea principal es definir una linea
     * recta
     */

    /* A continuación se define un vector para entrenamiento y prueba con
     * valor inicial de 1. */

    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());
    /* Se redimensiona las matrices para ubicarlas en el vector de
     * unos creado anteriormente. Similar a la función reshape de Numpy
     */
    X_train.conservativeResize(X_train.rows(),X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vectorTrain;

    X_test.conservativeResize(X_test.rows(),X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vectorTest;

    // Se define el vector theta
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());

    // Se define alpha como ratio	 de aprendizaje (salto)
    float alpha = 0.01;
    int iteraciones = 1000;

    // De igual forma se procederá a desempaquetar la tupla
    // Dada por el objeto (modelo)
    std::tuple<Eigen::VectorXd, std::vector<float>> gradiente = LR.GradDesc(X_train, y_train,theta,alpha,iteraciones);

    Eigen::VectorXd thetaSalida;
    std::vector<float> costo;
    std::tie(thetaSalida, costo) = gradiente;

    //Se imprime el vector de coeficientes o pesos
    std::cout << thetaSalida << std::endl;

    // Se imprime el vector de costo, para apreciar como decrementa su valor
    for (auto v: costo){
        std::cout << v << std::endl;
    }

    // Se exporta los valores de la funcion de costos y los coeficientes de theta a ficheros.
    /*extraer.conVectorFichero(costo, "VectorCosto.txt");
    extraer.EigenToFile(thetaSalida, "VectorTheta.txt");

    /* Se calcula de nuevo el promedio y la desviacion estandar basada en los datos
    para calcular  y_hat (predicciones). */
    auto promedioData = extraer.Promedio(MatDataFrame);
    auto numFeatures = promedioData(0,9);
    auto escalados = MatDataFrame.rowwise()-MatDataFrame.colwise().mean();
    auto sigmaData = extraer.DesvStand(escalados);
    auto sigmaFeatures = sigmaData(0, 9);

    Eigen::MatrixXd y_train_hat = (X_train*thetaSalida*sigmaFeatures).array() + numFeatures;
    Eigen::MatrixXd y = MatDataFrame.col(9).topRows(43153);

    /* A continuación se determina que tan bueno es nuestro modelo. */
    float R2 = LR.RSquared(y, y_train_hat);
    std::cout << "Metrica R2 " << R2 << std::endl;

    /* Se exporta y_train_hat a fichero*/
    extraer.EigenToFile(y_train_hat, "y_train_hat.txt");








    return EXIT_SUCCESS;














}
