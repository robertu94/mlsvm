ALL: mlsvm_classifier mlsvm_classifier_pressio
CC 	 = clang++ -L. 
CXX 	 = clang++ -L. 
CFLAGS 	 = -I.	
CPPFLAGS = $(shell pkg-config parmetis --cflags)
SANITIZE=
CXXFLAGS = -std=c++17 -g -Og -fno-omit-frame-pointer $(SANITIZE) -w -W -Wall -Weffc++ -Wextra -pedantic
LOCDIR   = .
MAIN 	 = mlsvm_classifier.cc
MANSEC   = Mat

LIBS= -lpugixml -lm -fopenmp $(shell pkg-config --libs parmetis) -lstdc++fs $(shell pkg-config --libs liblz4) $(shell pkg-config --libs flann) $(SANITIZE)
#LIBFLANN= /usr/local/lib/libflann_cpp_s.a
LIBFLANN= ${FLANN_PATH}

MLSVM_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc k_fold.cc svm_weighted.cc config_params.cc model_selection.cc solver.cc partitioning.cc refinement.cc  main_recursion.cc coarsening.cc loader.cc ds_node.cc ds_graph.cc mlsvm_classifier.cc mlsvm_main_core.cc
MLSVM_OBJS = $(MLSVM_SRCS:.cc=.o)

MLSVM_PRESSIO_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc k_fold.cc svm_weighted.cc config_params.cc model_selection.cc solver.cc partitioning.cc refinement.cc  main_recursion.cc coarsening.cc loader.cc ds_node.cc ds_graph.cc mlsvm_classifier_pressio.cc mlsvm_main_core.cc
MLSVM_PRESSIO_OBJS = $(MLSVM_PRESSIO_SRCS:.cc=.o)


MLSVM_SEP_TEST_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc k_fold.cc svm_weighted.cc config_params.cc model_selection.cc solver.cc partitioning.cc refinement.cc  main_recursion.cc coarsening.cc loader.cc ds_node.cc ds_graph.cc mlsvm_sep_test_classifier.cc
MLSVM_SEP_TEST_OBJS = $(MLSVM_SEP_TEST_SRCS:.cc=.o)

MlsvmNoTest_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc k_fold.cc svm_weighted.cc config_params.cc model_selection.cc solver.cc partitioning.cc refinement.cc  main_recursion.cc coarsening.cc loader.cc ds_node.cc ds_graph.cc mlsvmClassifierNoTest.cc
MlsvmNoTest_OBJS = $(MlsvmNoTest_SRCS:.cc=.o)

SLSVM_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc k_fold.cc svm.cc config_params.cc model_selection.cc solver.cc loader.cc ds_node.cc ds_graph.cc main_sl.cc
SLSVM_OBJS = $(SLSVM_SRCS:.cc=.o)

UT_SRCS= svm_weighted.cc solver.cc model_selection.cc ut_ms.cc ut_common.cc ut_kf.cc ut_partitioning.cc ds_node.cc ds_graph.cc coarsening.cc partitioning.cc ut_mr.cc pugixml.cc config_params.cc etimer.cc ut_cf.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc ut_cs.cc ut_ld.cc  ut_main.cc
#svm.cc  model_selection.cc main_recursion.cc coarsening.cc  ds_node.cc ds_graph.cc 
UT_OBJS = $(UT_SRCS:.cc=.o)

CV_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc ./tools/cross_validation.cc
CV_OBJS = $(CV_SRCS:.cc=.o)

SAT_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc svm_unweighted.cc solver.cc ./tools/single_svm_train.cc
SAT_OBJS = $(SAT_SRCS:.cc=.o)

SATIW_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc svm_weighted.cc solver.cc ./tools/single_svm_train_instance_weight.cc
SATIW_OBJS = $(SATIW_SRCS:.cc=.o)


SAP_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc svm_weighted.cc solver.cc ./tools/single_svm_predict.cc
SAP_OBJS = $(SAP_SRCS:.cc=.o)

PREDICT_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc svm_weighted.cc solver.cc ./tools/mlsvm_predict.cc
PREDICT_OBJS = $(PREDICT_SRCS:.cc=.o)

ZSCORE_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc preprocessor.cc ./tools/mlsvm_zscore.cc
ZSCORE_OBJS = $(ZSCORE_SRCS:.cc=.o)

CSV_PETSC_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc convertor.cc loader.cc k_fold.cc  ./tools/mlsvm_csv_petsc.cc
CSV_PETSC_OBJS = $(CSV_PETSC_SRCS:.cc=.o)

CSV_TO_VEC_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc convertor.cc loader.cc k_fold.cc  ./tools/mlsvm_csv_to_vec.cc
CSV_TO_VEC_OBJS = $(CSV_TO_VEC_SRCS:.cc=.o)


KNN_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc  ./tools/mlsvm_knn.cc
KNN_OBJS = $(KNN_SRCS:.cc=.o) $(LIBFLANN) 

PERS_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc svm_unweighted.cc solver.cc model_selection.cc personalized.cc personalized_main.cc
PERS_OBJS = $(PERS_SRCS:.cc=.o)

TestMatrix_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc ./tools/test_matrix.cc
TestMatrix_OBJS = $(TestMatrix_SRCS:.cc=.o)

ConvertTools_SRCS= etimer.cc common_funcs.cc ut_common.cc convertor.cc ./tools/convert_tools.cc
ConvertTools_OBJS = $(ConvertTools_SRCS:.cc=.o)

Convert_libsvm_PETSc_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc ut_common.cc loader.cc convertor.cc ./tools/mlsvm_libsvm_PETSc.cc
Convert_libsvm_PETSc_OBJS = $(Convert_libsvm_PETSc_SRCS:.cc=.o)

mlsvm_Save_knn_SRCS= etimer.cc common_funcs.cc loader.cc pugixml.cc OptionParser.cc config_params.cc k_fold.cc ./tools/mlsvm_save_knn.cc
mlsvm_Save_knn_OBJS = $(mlsvm_Save_knn_SRCS:.cc=.o)


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


info: 
	-@echo $(LIBS)
	-@echo $(MLSVM_OBJS)
#pugixml: g++ -c pugixml.cpp pugixml.hpp
pugixml: mpicxx -c pugixml.cpp pugixml.hpp


main_libs_inst:  etimer.o common_funcs.o OptionParser.o k_fold.o svm.o config_params.o model_selection.o solver.o partitioning.o refinement.o main_recursion.o coarsening.o loader.o ds_node.o ds_graph.o main.o chkopts
	-${CLINKER}  etimer.o common_funcs.o OptionParser.o k_fold.o svm.o config_params.o model_selection.o solver.o partitioning.o refinement.o main_recursion.o coarsening.o loader.o ds_node.o ds_graph.o main.o  ${PETSC_MAT_LIB} -o main $(LIBS)
	${RM} main.o 

mlsvm_classifier: $(MLSVM_OBJS) chkopts
	-${CLINKER} $(MLSVM_OBJS)  ${PETSC_MAT_LIB} ${LIBS} -o mlsvm_classifier

mlsvm_classifier_pressio: $(MLSVM_PRESSIO_OBJS) chkopts
	${CXXLINKER} $(MLSVM_PRESSIO_OBJS)  ${PETSC_MAT_LIB} ${LIBS} -o mlsvm_classifier_pressio
	
mlsvm_sep_test_classifier: $(MLSVM_SEP_TEST_OBJS) chkopts
	-${CLINKER} $(MLSVM_SEP_TEST_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_sep_test_classifier
	${RM} mlsvm_sep_test_classifier.o 

mlsvmNoTestClassifier: $(MlsvmNoTest_OBJS) chkopts
	-${CLINKER} $(MlsvmNoTest_OBJS)  ${PETSC_MAT_LIB} -o mlsvmNoTestClassifier
	${RM} mlsvmNoTestClassifier.o 

main_sl: $(SLSVM_OBJS) chkopts			# single level (no multi level which means no v-cycle)
	-${CLINKER} $(SLSVM_OBJS)  ${PETSC_MAT_LIB} -o slsvm 
	${RM} main_sl.o 

prepare_labels:  prepare_labels.o chkopts
	-${CLINKER}  prepare_labels.o  ${PETSC_VEC_LIB} -o prepare_labels 
	${RM} prepare_labels.o 
	
ut_main_libs_inst:  ut_mr.o main_recursion.o etimer.o model_selection.o svm.o loader.o ds_node.o ds_graph.o coarsening.o common_funcs.o config_params.o OptionParser.o ut_main.o chkopts
	-${CLINKER}  ut_mr.o  main_recursion.o etimer.o model_selection.o svm.o loader.o ds_node.o ds_graph.o coarsening.o common_funcs.o config_params.o OptionParser.o ut_main.o ${PETSC_MAT_LIB} -o ut_main -lpugixml
	${RM} ut_main.o 
	
ut_main: $(UT_OBJS) chkopts
	-${CLINKER} $(UT_OBJS)  ${PETSC_MAT_LIB} -o ut_main 
	${RM} ut_main.o 
	
main_test: $(MLSVM_OBJS) chkopts
	-${CLINKER}  $(MLSVM_OBJS)  ${PETSC_MAT_LIB} -o main $(LIBS)
	${RM} main.o 

cv: $(CV_OBJS) chkopts
	-${CLINKER} $(CV_OBJS)  ${PETSC_MAT_LIB} -o cv 
	${RM} cv.o 

sat_normal: $(SAT_OBJS) chkopts
	-${CLINKER} $(SAT_OBJS)  ${PETSC_MAT_LIB} -o sat_normal 
	${RM} sat_normal.o 
	
sat_weighted: $(SATIW_OBJS) chkopts
	-${CLINKER} $(SATIW_OBJS)  ${PETSC_MAT_LIB} -o sat_weighted 
	${RM} sat_weighted.o 


sap: $(SAP_OBJS) chkopts
	-${CLINKER} $(SAP_OBJS)  ${PETSC_MAT_LIB} -o sap 
	${RM} sap.o 
	
mlsvm_predict: $(PREDICT_OBJS) chkopts
	-${CLINKER} $(PREDICT_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_predict
	${RM} mlsvm_predict.o

mlsvm_zscore: $(ZSCORE_OBJS) chkopts
	-${CLINKER} $(ZSCORE_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_zscore
	${RM} mlsvm_zscore.o

mlsvm_csv_petsc: $(CSV_PETSC_OBJS) chkopts
	-${CLINKER} $(CSV_PETSC_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_csv_petsc
	${RM} mlsvm_csv_petsc.o

mlsvm_csv_to_vec: $(CSV_TO_VEC_OBJS) chkopts
	-${CLINKER} $(CSV_TO_VEC_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_csv_to_vec
	${RM} mlsvm_csv_to_vec.o

mlsvm_knn: $(KNN_OBJS) chkopts
	-${CLINKER}  $(KNN_OBJS) ${PETSC_MAT_LIB} ${LIBS} -o mlsvm_knn
	${RM} mlsvm_knn.o
	
pers: $(PERS_OBJS) chkopts
	-${CLINKER} $(PERS_OBJS)  ${PETSC_MAT_LIB} -o pers
	${RM} pers.o 

testmatrix: $(TestMatrix_OBJS) chkopts
	-${CLINKER} $(TestMatrix_OBJS)  ${PETSC_MAT_LIB} -o testmatrix
	${RM} testmatrix.o 

cvt: $(ConvertTools_OBJS) chkopts
	-${CLINKER} $(ConvertTools_OBJS)  ${PETSC_MAT_LIB} -o convert_tools
	${RM} convert_tools.o 

mlsvm_libsvm_petsc: $(Convert_libsvm_PETSc_OBJS) chkopts
	-${CLINKER} $(Convert_libsvm_PETSc_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_libsvm_petsc
	${RM} mlsvm_libsvm_petsc.o 
	
	
mlsvm_save_knn: $(mlsvm_Save_knn_OBJS) chkopts
	-${CLINKER} $(mlsvm_Save_knn_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_save_knn
	${RM} mlsvm_save_knn.o 

clean_main:	#PETSc Makefile has clean method which conflict with clean method here, therefore this one is created
	make clean
	rm mlsvm_classifier
	echo "clean is done"
