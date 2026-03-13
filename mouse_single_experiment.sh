#!/bin/bash

# Output directory for job files and logs
output_dir="job_outputs"
mkdir -p $output_dir

# Generate job name
job_name="mouse_training_VarRUOT"

# Create the submission script file
script_file="${output_dir}/${job_name}_job.bsub"
{
   echo "#!/bin/bash -l"
   echo ""
   echo "## Scheduler parameters ##"
   echo "#BSUB -J $job_name                    # job name"
   echo "#BSUB -o ${output_dir}/${job_name}.%J.stdout          # output written to specific file"
   echo "#BSUB -e ${output_dir}/${job_name}.%J.stderr          # errors written to specific file"
   echo "#BSUB -W 5:00                         # wallclock time"
   echo "#BSUB -n 1                            # number of CPU cores"
   echo "#BSUB -M 32000                        # amount of RAM in MB"
   echo '#BSUB -gpu "num=1"'
   echo "module load cuda/12.6"
   echo ""
   echo "## Job parameters ##"
   echo "uv run python -m main_mouse"
} > "$script_file"

# Submit the job using bsub
bsub < $script_file

# Optionally, print a message
echo "Submitted job: $job_name"
