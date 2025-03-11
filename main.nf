#!/usr/bin/env nextflow


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//include { MOLFLOW  } from './workflows/molflow'
include { RUNDATA    } from './workflows/flow'

// include { PIPELINE_INITIALISATION } from './subworkflows/local/utils_nfcore_molflow_pipeline'
// include { PIPELINE_COMPLETION     } from './subworkflows/local/utils_nfcore_molflow_pipeline'
// include { getGenomeAttribute      } from './subworkflows/local/utils_nfcore_molflow_pipeline'

/*



// TODO nf-core: Remove this line if you don't need a FASTA file
//   This is an example of how to use getGenomeAttribute() to fetch parameters
//   from igenomes.config using `--genome`
params.fasta = getGenomeAttribute('fasta')

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOWS FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Run main analysis pipeline depending on type of input
//
// workflow OPENBIO_MOLFLOW {

//     take:
//     samplesheet // channel: samplesheet read in from --input

//     main:

//     //
//     // WORKFLOW: Run pipeline
//     //
//     MOLFLOW (
//         samplesheet
//     )
//     emit:
//     multiqc_report = MOLFLOW.out.multiqc_report // channel: /path/to/multiqc_report.html
// }
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow {

    main:
    //
    // SUBWORKFLOW: Run initialisation tasks
    //
    // PIPELINE_INITIALISATION (
    //     params.version,
    //     params.validate_params,
    //     params.monochrome_logs,
    //     args,
    //     params.outdir,
    //     params.input
    // )
    Channel
        .fromPath('/home/bio-17/projects/drug/nf_drug/nfdrug/codes/BFregNN-Cox-for-pyroptosis-in-TNBC/data/9_drug_targets_1.0_revised.tsv')
        .splitCsv(header: true, sep: '\t')
        .set { drug_targets }

    drug_targets.view()


    //
    // WORKFLOW: Run main workflow
    //
    // OPENBIO_MOLFLOW (
    //     PIPELINE_INITIALISATION.out.samplesheet
    // )
    //
    // SUBWORKFLOW: Run completion tasks
    //
    // PIPELINE_COMPLETION (
    //     params.email,
    //     params.email_on_fail,
    //     params.plaintext_email,
    //     params.outdir,
    //     params.monochrome_logs,
    //     params.hook_url,
    //     OPENBIO_MOLFLOW.out.multiqc_report
    // )
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
