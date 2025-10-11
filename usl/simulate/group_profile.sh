BS_LIST=(4 8)
for BS in "${BS_LIST[@]}"; do
    echo "BS=$BS"
    bash usl/simulate/profile.sh 300 $BS meta-llama/llama3.2-1b 
done
