from astroquery.simbad import Simbad
#results= Simbad.query_object("GAIA DR2 604907012195819648")
#print(result_table)
#print(dir(results))
#['Column', 'ColumnClass', 'MaskedColumn', 'Row', 'TableColumns', 'TableFormatter', '__array__', '__byteinit_indices', '_ipython_key_completions_', '_is_list_or_tuple_of_str', '_make_index_row_display_table', '_make_table_from_cols', '_mask', '_masked', '_meta', '_new_from_slice', '_replace_cols', '_replace_column_warnings', '_repr_html_', '_set_masked', '_set_masked_from_cols', '_set_row', 'add_column', 'add_columns', 'add_index', 'add_row', 'argsort', 'as_array', 'colnames', 'columns', 'convert_bytestring_to_unicode', 'convert_unicode_to_bytestring', 'copy', 'dtype', 'errors', 'field', 'filled', 'formatter', 'from_pandas', 'group_by', 'groups', 'has_mixin_columns', 'iloc', 'index_column', 'index_mode', 'indices', 'info', 'insert_row', 'itercols', 'keep_columns', 'keys', 'loc', 'loc_indices', 'mask', 'masked', 'meta', 'more', 'pformat', 'pformat_all', 'pprint', 'pprint_all', 'primary_key', 'read', 'remove_column', 'remove_columns', 'remove_indices', 'remove_row', 'remove_rows', 'rename_column', 'rename_columns', 'replace_column', 'reverse', 'show_in_browser', 'show_in_notebook', 'sort', 'to_pandas', 'write']
#'MAIN_ID','RA','DEC','RA_PREC','DEC_PREC','COO_ERR_MAJA','COO_ERR_MINA','COO_ERR_ANGLE','COO_QUAL','COO_WAVELENGTH','COO_BIBCODE'

#Sim = Simbad()
#print(dir(Sim))


#['_VOTABLE_FIELDS', 'add_votable_fields', , 'get_field_description', 'get_votable_fields', 'list_votable_fields','remove_votable_fields', 'reset_votable_fields']
'''
1. The parameter filtername must correspond to an existing filter. Filters include: B,V,R,I,J,K.  They are checked by SIMBAD but not astroquery.simbad

2. Fields beginning with rvz display the data as it is in the database. Fields beginning with rv force the display as a radial velocity. Fields beginning with z force the display as a redshift

3. For each measurement catalog, the VOTable contains all fields of the first measurement. When applicable, the first measurement is the mean one. 
'''
#Available VOTABLE fields: bibcodelist(y1-y2), biblio, cel, cl.g, coo(opt), coo_bibcode,coo_err_angle,coo_err_maja,coo_err_mina,coo_qual,coo_wavelength,coordinates,dec(opt),dec_prec,diameter,dim,dim_angle,dim_bibcode,dim_incl,dim_majaxis,dim_minaxis,dim_qual,dim_wavelength,dimensions,distance,distance_result,einstein,fe_h,flux(filtername),flux_bibcode(filtername),flux_error(filtername),flux_name(filtername),flux_qual(filtername),flux_system(filtername),flux_unit(filtername),fluxdata(filtername),gcrv,gen,gj,hbet,hbet1,hgam,id(opt),ids,iras,irc,iso,iue,jp11,link_bibcode,main_id,measurements,membership,mesplx,mespm,mk,morphtype,mt,mt_bibcode,mt_qual,otype,otype(opt),otypes,parallax,plx,plx_bibcode,plx_error,plx_prec,plx_qual,pm,pm_bibcode,pm_err_angle,pm_err_maja,pm_err_mina,pm_qual,pmdec,pmdec_prec,pmra,pmra_prec,pos,posa,propermotions,ra(opt),ra_prec,rot,rv_value,rvz_bibcode,rvz_error,rvz_qual,rvz_radvel,rvz_type,rvz_wavelength,sao,sp,sp_bibcode,sp_nature,sp_qual,sptype,td1,typed_id,ubv,uvby,uvby1,v*,velocity,xmm,z_value,
#For more information on a field:
#Simbad.get_field_description ('field_name')

#Available VOTABLE fields:z_value,
#cel = Celescope catalog of ultra-violet photometry
#coordinates = all fields related with coordinates
#distance = Measure of distances by several means
#fe_h = Stellar parameters (Teff, log(g) and [Fe/H]) taken from the literature.
#parallax = all fields related to parallaxes
#plx = parallax value
#flux(filtername) --> "flux(b)"
#flux_error(filtername) --> "flux_error(b)"
#Simbad.get_field_description("")

Simbad.add_votable_fields("distance","parallax","fe_h","flux(B)","flux_error(B)","flux(V)","flux_error(V)")
#print(Simbad.get_votable_fields())
results = Simbad.query_object("GAIA DR2 604921821242945536")
#results.show_in_browser()
#'show_in_browser', 'show_in_notebook'
#print(results.keys())
#print(results['Fe_H_Teff'])
#print(dir(results['Fe_H_Teff']))

#http://simbad.u-strasbg.fr/simbad/sim-id?Ident=%401128178&Name=NGC++2682++++84&submit=display+all+measurements#lab_meas
#https://simbad.u-strasbg.fr/simbad/sim-id?Ident=GAIA+DR2+604907012195819648&NbIdent=1&Radius=2&Radius.unit=arcmin&submit=submit+id
#https://simbad.u-strasbg.fr/simbad/sim-id?mescat.distance=on&mescat.velocities=on&mescat.rot=on&mescat.pm=on&mescat.fe_h=on&mescat.plx=on&mescat.v*=on&Ident=%401133098&Name=NGC++2682+++289&submit=display+all+measurements#lab_meas
star_name = str(results['MAIN_ID'])
star_name = star_name.split("\n")[-1].split(' ')
star_name = list(filter(None, star_name))
#print(star_name)
a = "a"
b = "b"
for i in star_name:
          try:
                    if a == "a":
                              a = str(int(i))
                    else:
                              b = str(int(i))
          except:
                    pass
#print(a,b)
#http://simbad.u-strasbg.fr/simbad/sim-id?Ident=%401128235&Name=NGC%20%202682%20%20%20128&submit=submit
#http://simbad.u-strasbg.fr/simbad/sim-id?Ident=%401128235&Name=NGC++2682+++128&submit=display+all+measurements#lab_meas

#http://simbad.u-strasbg.fr/simbad/sim-id?Ident=%401130581&Name=Cl*%20NGC%202682%20%20%20SAND%20%20%20%20%20859&submit=submit
#http://simbad.u-strasbg.fr/simbad/sim-id?Ident=%401130581&Name=Cl*+NGC+2682+++SAND+++++859&submit=display+all+measurements#lab_meas
