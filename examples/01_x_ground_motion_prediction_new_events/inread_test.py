import gmacc.gmeval.inout as GMio

file = 'events/iris/event.xml'

src = GMio.pyrocko_to_source(file)
# src = GMio.pyrocko2_to_source(file)
# src = GMio.obspy_to_source(file)

print(src)