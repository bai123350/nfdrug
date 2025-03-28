
process DATATETCH {
    tag "$meta.id"
    label 'process_medium'

    conda "${moduleDir}/environment.yml"
    
    // container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
    //     'docker://kjd12/pyhtonbio:1.0' :
    //     'kjd12/pyhtonbio:1.0' }"  

    
    publishDir "${params.outdir}/datafetch", mode: 'copy'

    input:
    tuple val(meta), val(reads)

    output:
    val(meta),  emit: meta_id
    path("*.json"), emit: json
    // tuple val(meta), path("*.zip") , emit: zip
    // path  "versions.yml"           , emit: versions

    script:
    """
    fetch.py --score ${params.score} --out "res.json" --path1 ${reads[0]} --path2 ${reads[1]} 
    """

    stub:
    """
    touch res.json 
    """
}






