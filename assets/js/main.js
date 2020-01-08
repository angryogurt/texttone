$(document).ready(function(){
$('#form').submit(function(event){
    $("#in2").text("Проводим анализ тональности текста...");
    $.post("cgi-bin/input.py",{data:$("#in1").val()},onResponse);
    return false;
})
function onResponse(data){
    $("#in2").text("Текст позитивный с вероятностью в "+data+"%");
}
})