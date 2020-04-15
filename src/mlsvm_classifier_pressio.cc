#include <iostream>
#include <functional>
#include <algorithm>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include "mlsvm_main_core.h"
#include "config_params.h"
#include <getopt.h>

using namespace std::literals;
namespace fs = std::filesystem;

static void run_mlsvm(bool verbose, std::ostream& out, std::vector<const char*>& args, fs::path const& logfile_path) {
	int logfile = open(logfile_path.c_str(), O_WRONLY|O_CREAT);
  int devnull = open("/dev/null", O_WRONLY);
  int stdout_cpy = dup(STDOUT_FILENO);
  int stderr_cpy = dup(STDERR_FILENO);

  //redirect file descriptors while running the app
  if(verbose) {
    dup2(STDERR_FILENO, STDOUT_FILENO);
  } else {
    dup2(logfile, STDOUT_FILENO);
    dup2(logfile, STDERR_FILENO);
  }
  mlsvm_main(args.size(), &args[0]);


  //ensure that output is flushed, otherwise buffered
  //output will be written after the file descriptors
  //restored
  fflush(stdout);
  fflush(stderr);

  //restore output prior to printing
  dup2(stdout_cpy, STDOUT_FILENO);
  dup2(stderr_cpy, STDERR_FILENO);

  auto results = paramsInst->get_final_results();
  out << "external:api=3" << std::endl;
  out << "acc=" << results.acc << std::endl;
  out << "sn=" << results.sn << std::endl;
  out << "sp=" << results.sp << std::endl;
  out << "ppv=" << results.ppv << std::endl;
  out << "npv=" << results.npv << std::endl;
  out << "f1=" << results.f1 << std::endl;
  out << "gm=" << results.gm << std::endl;
  out  << "iters=" << (paramsInst->get_main_num_kf_iter() * paramsInst->get_main_num_repeat_exp()) << std::endl;


  //release the file descriptors
  close(devnull);
  close(stdout_cpy);
  close(stderr_cpy);
	close(logfile);
}

static void print_exemplars(std::ostream& out) {
  out << R"(external:api=3
acc=0.001
sn=0.001
sp=0.001
ppv=0.001
npv=0.001
f1=0.001
gm=0.001
iters=1)" << std::endl;
}

/**
 * map containing the options to pass down to mlsvm
 */
std::map<const std::string, const std::vector<const char*>> const ARGS {
  {"advertisement", {"-r", "1", "--cs_we", "0.05", "--rf_2nd", "0", "--pr_start", "5000", "--pr_max", "1000", "-s", "10"}},
  {"buzz", {"-r", "2", "--cs_we", "0.05", "--rf_2nd", "0", "--pr_start", "5000", "--pr_max", "1000", "-s", "100"}},
  {"clean", {"-r", "1", "--cs_we", "0.001", "--rf_2nd", "1", "--pr_start", "5000", "--pr_max", "1000", "-s", "100"}},
  {"cod", {"-r", "2", "--cs_we", "0.01", "--rf_2nd", "0", "--pr_start", "3000", "--pr_max", "1000", "-s", "100"}},
  {"forest", {"-r", "1", "--cs_we", "0.001", "--rf_2nd", "0", "--pr_start", "5000", "--pr_max", "1000", "-s", "10"}},
  {"letter", {"-r", "1", "--cs_we", "0.05", "--rf_2nd", "1", "--pr_start", "3000", "--pr_max", "1500", "-s", "100"}},
  {"ringnorm", {"-r", "1", "--cs_we", "0.05", "--rf_2nd", "0", "--pr_start", "5000", "--pr_max", "1000", "-s", "100"}},
  {"twonorm", {"-r", "1", "--cs_we", "0.01", "--rf_2nd", "0", "--pr_start", "10000", "--pr_max", "1000", "-s", "100"}},
  {"susy", {"-r", "2", "--cs_we", "0.05", "--rf_2nd", "0", "--pr_start", "5000", "--pr_max", "1000", "-s", "10"}},
  {"higgs", {"-r", "1", "--cs_we", "0.05", "--rf_2nd", "0", "--pr_start", "5000", "--pr_max", "1000", "-s", "10"}}
};

/**
 * structure for holding pressio metrics options
 */
struct cmdline_args {
  std::string dataset;
  fs::path decompressed;
  fs::path datadir;
  bool verbose = false;
  bool print_exemplars = true;
};

/**
 * parse options from the command line
 */
cmdline_args parse_args(int argc, char* const  argv[]) {
  static struct option long_options[] = {
    {"dim", required_argument, 0, 0},
    {"type", required_argument, 0, 0},
    {"input", required_argument, 0, 0},
    {"decompressed", required_argument, 0, 0},
    {"external_dataset", required_argument, 0, 0},
    {"external_datadir", required_argument, 0, 0},
    {"external_verbose", no_argument, 0, 0},
    {"api", required_argument, 0, 0}
  };

  cmdline_args result;
  while(true) {
    int option_index;
    int c = getopt_long(argc, argv, "", long_options, &option_index);
    if(c==-1) break;
    switch(c) {
      case 0:
        {
          switch(option_index) {
            case 0:
            case 1:
            case 2:
              result.print_exemplars = false;
              break;
            case 3: //decompressed
              result.decompressed = optarg;
              result.print_exemplars = false;
              break;
            case 4: //external_dataset
              result.dataset = optarg;
              break;
            case 5:
              result.datadir = optarg;
              break;
            case 6:
              result.verbose = true;
              break;
            default: 
              //we're going to ignore the other optsion since data is passed in petsc format
              break;
          }
          break;
        }
    }
  }

  return result;
}

/**
 * An RAII wrapper for copying files
 */
struct tmp_link {
  tmp_link(fs::path const& src, fs::path const& dst): filename(dst) {
    fs::copy_file(src, dst);
  }
  ~tmp_link() {
    fs::remove(filename);
  }
  tmp_link(tmp_link const&)=delete;
  tmp_link& operator=(tmp_link const&)=delete;

  tmp_link(tmp_link &&)=default;
  tmp_link& operator=(tmp_link &&)=default;

  private:
  fs::path filename;
};

/**
 * An RAII wrapper for a tmp directory
 */
struct tmp_dir {
  tmp_dir() noexcept= default;
  tmp_dir(fs::path const& name): filename(name) {
    fs::create_directory(name);
  }
  ~tmp_dir() {
    if(!filename.empty()) fs::remove_all(filename);
  }
  tmp_dir(tmp_dir const&)=delete;
  tmp_dir& operator=(tmp_dir const&)=delete;

  tmp_dir(tmp_dir &&)noexcept=default;
  tmp_dir& operator=(tmp_dir &&)noexcept=default;

  private:
  fs::path filename;
};

int main(int argc, char *argv[])
{
  int rank;
  MPI_Comm petsc_comm;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //mlsvm expects to be called from a MPI comm with size==1
  //so create one here
  MPI_Comm_split(MPI_COMM_WORLD, rank == 0, rank, &petsc_comm);

  auto cmdline_args = parse_args(argc, argv);
  std::ostringstream out;
  
  if(cmdline_args.print_exemplars) {
    print_exemplars(out);
  } else {
    if(cmdline_args.decompressed.empty()) {
      if(rank == 0) {
        std::cerr << "failed to load decompressed data " << cmdline_args.decompressed << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string filename = cmdline_args.decompressed.filename();
    filename = filename.substr(0, filename.rfind("_zsc_data.dat"));

    fs::path dirname = cmdline_args.decompressed.parent_path();
    

    std::vector<tmp_link> tmp_files;
    fs::path tmp_path = cmdline_args.decompressed.string() + ".tmp_dir"s;
    tmp_dir temp_dir;
    if(rank == 0) {
      //copy the auxiliary files
      std::vector<std::string> aux_files = {"_label.dat", "_maj_norm_data.dat",
                     "_maj_norm_data_dists.dat",
                     "_maj_norm_data_indices.dat",
                     "_min_norm_data.dat",
                     "_min_norm_data_dists.dat",
                     "_min_norm_data_indices.dat"};
      for (auto const& aux_file : aux_files) {
        try {
        tmp_files.emplace_back(
            cmdline_args.datadir / (cmdline_args.dataset + aux_file),
            dirname / (filename + aux_file)
          );
        } catch(fs::filesystem_error const&) {
          //deliberately ignore std::filesystem_errors that occur when
          //the src and dest are the same
        }
      }
      temp_dir = {tmp_path};
    }

    //finish parsing arguments doing this here to retain scope
    std::vector<const char*> args;
    args.push_back("mlsvm_classifier");
    auto& dataset_args = ARGS.at(cmdline_args.dataset);
    std::copy(std::begin(dataset_args), std::end(dataset_args), std::back_inserter(args));
    args.push_back("-f");
    args.push_back(filename.c_str());
    args.push_back("--ds_p");
    args.push_back(dirname.c_str());
    args.push_back("--tmp_p");
    args.push_back(tmp_path.c_str());
		fs::path logfile_path = cmdline_args.datadir / (cmdline_args.dataset + std::string(cmdline_args.decompressed.filename()) + ".log");


    if(rank == 0) {
      //if you are only running petsc on a subcommunicator
      //accodring to the docs for PETSc you should set PETSC_COMM_WORLD
      PETSC_COMM_WORLD = petsc_comm;
      run_mlsvm(cmdline_args.verbose, out, args, logfile_path);
      //cleanup the tmpdir
    }

  }

  MPI_Comm parent;
  MPI_Comm_get_parent(&parent);
  if(rank == 0 && parent == MPI_COMM_NULL) {
    std::cout << out.str();
    std::flush(std::cout);
  } else {
    const int parent_rank = 0;
    const int parent_tag = 0;

    int status_code = 0;
    std::string stdout_str = out.str();
    int stdout_len = stdout_str.size();
    std::string stderr_str;
    int stderr_len = stderr_str.size();

    MPI_Send(&status_code, 1, MPI_INT, parent_rank, parent_tag, parent);
    MPI_Send(&stdout_len, 1, MPI_INT, parent_rank, parent_tag, parent);
    MPI_Send(stdout_str.c_str(), stdout_len, MPI_CHAR, parent_rank, parent_tag, parent);
    MPI_Send(&stderr_len, 1, MPI_INT, parent_rank, parent_tag, parent);
    MPI_Send(stderr_str.c_str(), stderr_len, MPI_CHAR, parent_rank, parent_tag, parent);
  }

  MPI_Comm_free(&petsc_comm);
  MPI_Finalize();
  return 0;
}
