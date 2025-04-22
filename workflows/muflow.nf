/**
 * @author: @KONG jin dong
 * @date: 2025-03-01
**/
include { DATATETCH        } from '../modules/local/datafetch/main'
include { MUTILDATAPROCESS } from '../modules/local/dataprocess/mudrug'
include { MUALLMODELS      } from '../modules/local/models/mumodel'
include { MUTRAIN          } from '../modules/local/train/mutrain'
include { VISIABLE         } from '../modules/local/visiable/main'
include { VISSHAP          } from '../modules/local/visshap/main'




workflow MURUNDATA {
    take:
    samplesheet // channel: samplesheet read in from --input

    main:
    // WORKFLOW: Run pipeline
    ch_versions = Channel.empty()
    ch_multiqc_files = Channel.empty()

    // MODULE: Run DATATETCH
    DATATETCH(
        samplesheet
    )

    MUTILDATAPROCESS(
        samplesheet,
        DATATETCH.out.json,
    )

    MUALLMODELS(
        samplesheet,
        MUTILDATAPROCESS.out.all_folders
    )

    MUTRAIN(
        samplesheet,
        MUALLMODELS.out.model_dirs,
    )

    VISIABLE(
        samplesheet,
        MUTRAIN.out.train_dirs,
    )

    VISSHAP(
        samplesheet,
        DATATETCH.out.json,
        MUTILDATAPROCESS.out.all_folders,
        MUALLMODELS.out.model_dirs,
        MUTRAIN.out.train_dirs,
    )

    vis_files = getBaseFilenames(VISSHAP.out.pdf)

    // 筛选匹配的文件夹
    matched_folders = filterMatchingFolders(MUTILDATAPROCESS.out.all_folders, vis_files)
    train_folders = filterMatchingFolders(MUTRAIN.out.train_dirs, vis_files)

    // 混合通道
    ch_multiqc_files = mixChannels(
        ch_multiqc_files,
        DATATETCH.out.json,
        matched_folders,
        train_folders,
        VISIABLE.out.pdf,
        VISSHAP.out.pdf
    )

    println("ch_multiqc_files: ${ch_multiqc_files.view()}")



    // emit:
    // json_report = DATATETCH.out.json // channel: /path/to/multiqc_report.html
}


def getBaseFilenames(channel) {
    return channel
        .map { it ->
            def fname = it.toString().split('/')[-1]
            def base = fname.substring(0, fname.lastIndexOf('.'))
            return base
        }
        .collect()
        .map { files -> files as Set }
        .toList()
}

def filterMatchingFolders(folders_channel, vis_files) {
    return folders_channel
        .map { folders ->
            def folder_list = folders instanceof Collection ? folders : [folders]
            def vis_set = vis_files.val[0]
            return folder_list.findAll { folder ->
                def fname = folder.toString().split('/')[-1]
                return vis_set.contains(fname)
            }
        }
        .flatten()
        .unique()
}

def mixChannels(ch_multiqc_files, datatetch_json, matched_folders, train_folders, visiable_pdf, visshap_pdf) {
    return ch_multiqc_files.mix(
        datatetch_json.collect().ifEmpty([]),
        matched_folders.collect().ifEmpty([]),
        train_folders.collect().ifEmpty([]),
        visiable_pdf.collect().ifEmpty([]),
        visshap_pdf.collect().ifEmpty([])
    )
}
