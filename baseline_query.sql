select oh.slide_id, oh.frame_id, oh.cell_id, oh.x, oh.y, oh.type
 from ocular_hitlist oh 
  join slide on slide.slide_id = split_part(oh.slide_id, '_',1)
  join staining_batch sb on sb.staining_batch_id = slide.staining_batch_id
  join protocol p on p.protocol_id = sb.protocol_id
   where p.name = 'Baseline'
   order by oh.slide_id, oh.frame_id, oh.cell_id