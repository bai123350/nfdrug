process VISSHAP {
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/visshap", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(trainfoler)

    output:
    // path("*pdf") , emit : pdf
    path "versions.yml", emit: versions

    script:
    def prefix = task.ext.prefix ?: "shap"
    """
    shap.py --dir ${trainfoler}

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
