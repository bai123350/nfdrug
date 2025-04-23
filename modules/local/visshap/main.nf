process VISSHAP {
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/visshap", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(json, stageAs : "result.json")
    path(allfolders, stageAs: "allfolders/*")
    path(modeldirs, stageAs: "modelfolder/*")
    path(trainfoler, stageAs: "trainfolder/*")

    output:
    path("*pdf") , emit : pdf
    path("*csv") , emit : csv
    path "versions.yml", emit: versions

    script:
    def prefix = task.ext.prefix ?: "shap"
    def folders = trainfoler instanceof List ? trainfoler.join(',') : [trainfoler].join(',')
    def allf = allfolders instanceof List ? allfolders.join(',') : [allfolders].join(',')
    def model = modeldirs instanceof List ? modeldirs.join(',') : [modeldirs].join(',')
    """
    shap.py --dir ${folders} --all ${allf} --model ${model} --gene ${reads[5]}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
        torch: \$(python -c "import torch; print(torch.__version__,torch.cuda.is_available())")
        seaborn: \$(python -c "import seaborn; print(seaborn.__version__)")
    END_VERSIONS
    """


    stub:
    def prefix = task.ext.prefix ?: "shap"
    """
    touch "${prefix}_shap.pdf"
    touch "versions.yml"
    """

}
