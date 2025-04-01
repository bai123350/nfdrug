process  MUTILDATAPROCESS {
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://kjd12/pyhtonbio:1.0' :
        'kjd12/pyhtonbio:1.0' }"

    publishDir "${params.outdir}/mudataprocess", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(json, stageAs : "result.json")


    output:
    val(meta),  emit: meta_id
    path("drug/*"), emit: all_folders
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: "${meta.id}_process"

    """
    muprocess.py --path ${reads[2]}  --path1 ${reads[3]} --path2 ${reads[5]} --json ${json}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        networkx:  \$(python -c 'import networkx; print(networkx.__version__)')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}_process"
    """
    mkdir -p drug
    touch "${prefix}_drug.json"
    touch "${prefix}_tranfer.npz"
    touch "${prefix}_basic.json"
    touch "versions.yml"
    """
}
