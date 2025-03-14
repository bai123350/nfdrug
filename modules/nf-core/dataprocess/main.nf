

process DATAPROCESS {

    label 'process_medium'

    conda "${moduleDir}/environment.yml"
    
    // container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
    //     'docker://kjd12/pyhtonbio:1.0' :
    //     'kjd12/pyhtonbio:1.0' }"  

    
    publishDir "${params.outdir}/dataprocess", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(json)

    output:
    val(meta),  emit: meta_id
    path("*.json"), emit: json

    script:
    """
    process.py --path ${reads[2]}  --out "drug.json"
    """

    stub:
    """
    touch "drug.json"
    
    """
}