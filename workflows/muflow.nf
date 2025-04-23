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

/**
 other modules
**/
include { softwareVersionsToYAML } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { paramsSummaryMap       } from 'plugin/nf-schema'
include { paramsSummaryMultiqc   } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { MULTIQC                } from '../modules/nf-core/multiqc/main'
include { methodsDescriptionText } from '../subworkflows/local/utils_nfcore_molflow_pipeline'


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

    // println("ch_multiqc_files: ${ch_multiqc_files.view()}")

    // 收集所有模块的版本信息
    ch_versions = ch_versions.mix(
        MUTILDATAPROCESS.out.versions,
        MUALLMODELS.out.versions,
        MUTRAIN.out.versions,
        VISIABLE.out.versions,
        VISSHAP.out.versions
    )

    // println("ch_versions: ${ch_versions.view()}")

    //
    // Collate and save software versions
    //
    softwareVersionsToYAML(ch_versions)
        .collectFile(
            storeDir: "${params.outdir}/pipeline_info",
            name: 'pipeline_software_mqc_versions.yml',
            sort: true,
            newLine: true,
        )
        .set { ch_collated_versions }

     //
    // MODULE: MultiQC
    //
    // 读取默认的MultiQC配置文件，定义报告的基本布局和设置
    ch_multiqc_config = Channel.fromPath(
        "${projectDir}/assets/multiqc_config.yml",
        checkIfExists: true
    )

    // 允许用户提供自定义配置文件，覆盖默认设置
    ch_multiqc_custom_config = params.multiqc_config
        ? Channel.fromPath(params.multiqc_config, checkIfExists: true)
        : Channel.empty()

    // 允许用户提供自定义log，覆盖默认设置
    ch_multiqc_logo = params.multiqc_logo
        ? Channel.fromPath(params.multiqc_logo, checkIfExists: true)
        : Channel.empty()

    // 使用 paramsSummaryMap 函数生成工作流参数摘要
    // 参数包括 workflow 对象和 schema 文件路径
    summary_params = paramsSummaryMap(
        workflow,
        parameters_schema: "nextflow_schema.json"
    )

    // println("summary_params: ${summary_params}")

    // 创建一个值为参数摘要的通道，用于 MultiQC 报告
    ch_workflow_summary = Channel.value(paramsSummaryMultiqc(summary_params))

    // 将工作流摘要文件混入 multiqc 文件通道
    // collectFile 操作符将内容收集到指定名称的文件中
    ch_multiqc_files = ch_multiqc_files.mix(
        ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml')
    )

     // 三元运算符检查是否提供了自定义方法描述文件
    // 如果没有则使用默认模板文件
    ch_multiqc_custom_methods_description = params.multiqc_methods_description
        ? file(params.multiqc_methods_description, checkIfExists: true)
        : file("${projectDir}/assets/methods_description_template.yml", checkIfExists: true)

    // 创建包含方法描述文本的通道
    ch_methods_description = Channel.value(
        methodsDescriptionText(ch_multiqc_custom_methods_description)
    )

    // 将版本信息文件混入 multiqc 文件通道
    ch_multiqc_files = ch_multiqc_files.mix(ch_collated_versions)

    // 将方法描述文件混入 multiqc 文件通道
    // sort:true 表示对内容进行排序
    ch_multiqc_files = ch_multiqc_files.mix(
        ch_methods_description.collectFile(
            name: 'methods_description_mqc.yaml',
            sort: true,
        )
    )

    // 调用 MULTIQC 流程
    // collect() 和 toList() 操作符将通道内容转换为列表
    MULTIQC(
        ch_multiqc_files.collect(),
        ch_multiqc_config.toList(),
        ch_multiqc_custom_config.toList(),
        ch_multiqc_logo.toList(),
        [],
        [],
    )

    emit:
    multiqc_report = MULTIQC.out.report.toList() // 输出 MultiQC HTML 报告路径
    versions = ch_versions // 输出软件版本信息文件路径
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
