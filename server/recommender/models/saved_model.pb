??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
n
identifiersVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameidentifiers
g
identifiers/Read/ReadVariableOpReadVariableOpidentifiers*
_output_shapes
:*
dtype0
p

candidatesVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name
candidates
i
candidates/Read/ReadVariableOpReadVariableOp
candidates*
_output_shapes

:@*
dtype0
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_nameembedding/embeddings
}
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes

:@*
dtype0
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name113*
value_dtype0	
}
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_48*
value_dtype0	
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_2Const*
_output_shapes
:*
dtype0*v
valuemBkB17B23B13B19B9B11B3B22B18B16B24B14B26B12B7B21B15B10B8B25B2B1B27B5B4B6B20
?
Const_3Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                        	       
                                                                                                                              
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_2Const_3*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *"
fR
__inference_<lambda>_1962
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *"
fR
__inference_<lambda>_1967
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?
Const_4Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
query_model
identifiers
_identifiers

candidates
_candidates
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
trainable_variables
	variables
regularization_losses
	keras_api
GE
VARIABLE_VALUEidentifiers&identifiers/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE
candidates%candidates/.ATTRIBUTES/VARIABLE_VALUE

0

1
2
3
 
?
layer_metrics
metrics
trainable_variables
non_trainable_variables
	variables

layers
layer_regularization_losses
regularization_losses
 
3
lookup_table
token_counts
	keras_api
b

embeddings
trainable_variables
	variables
regularization_losses
	keras_api

0

1
 
?
layer_metrics
metrics
trainable_variables
non_trainable_variables
	variables

layers
 layer_regularization_losses
regularization_losses
ZX
VARIABLE_VALUEembedding/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
 
 

1
2

0
 

!_initializer
JH
table?query_model/layer_with_weights-0/token_counts/.ATTRIBUTES/table
 

0

0
 
?
"layer_metrics
#metrics
$non_trainable_variables
trainable_variables
	variables

%layers
&layer_regularization_losses
regularization_losses
 
 
 

	0

1
 
 
 
 
 
 
 
r
serving_default_input_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_1
hash_tableConstembedding/embeddings
candidatesidentifiers*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_1652
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameidentifiers/Read/ReadVariableOpcandidates/Read/ReadVariableOp(embedding/embeddings/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1Const_4*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_2012
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameidentifiers
candidatesembedding/embeddingsMutableHashTable*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_2034??
?

?
*__inference_brute_force_layer_call_fn_1686
queries
unknown
	unknown_0	
	unknown_1:@
	unknown_2:@
	unknown_3:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_15042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_1457
string_lookup_input
unknown
	unknown_0	
	unknown_1:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_14372
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1479
string_lookup_input<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_1475:@
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlestring_lookup_input9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1475*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_13912#
!embedding/StatefulPartitionedCall?
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1743
queriesG
Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleH
Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value	<
*sequential_embedding_embedding_lookup_1727:@0
matmul_readvariableop_resource:@
gather_resource:

identity_1

identity_2??Gather?MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?6sequential/string_lookup/None_Lookup/LookupTableFindV2?
6sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handlequeriesDsequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????28
6sequential/string_lookup/None_Lookup/LookupTableFindV2?
!sequential/string_lookup/IdentityIdentity?sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2#
!sequential/string_lookup/Identity?
%sequential/embedding/embedding_lookupResourceGather*sequential_embedding_embedding_lookup_1727*sequential/string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*=
_class3
1/loc:@sequential/embedding/embedding_lookup/1727*'
_output_shapes
:?????????@*
dtype02'
%sequential/embedding/embedding_lookup?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/1727*'
_output_shapes
:?????????@20
.sequential/embedding/embedding_lookup/Identity?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@22
0sequential/embedding/embedding_lookup/Identity_1?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOp?
MatMulMatMul9sequential/embedding/embedding_lookup/Identity_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identityn

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1p

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_2?
NoOpNoOp^Gather^MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup7^sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2p
6sequential/string_lookup/None_Lookup/LookupTableFindV26sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1468
string_lookup_input<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_1464:@
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlestring_lookup_input9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1464*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_13912#
!embedding/StatefulPartitionedCall?
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
__inference_save_fn_1946
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2A
?MutableHashTable_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1Q
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const\

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:2

Identity_2W

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1^

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:2

Identity_5?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?	
?
C__inference_embedding_layer_call_and_return_conditional_losses_1894

inputs	'
embedding_lookup_1888:@
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_1888inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/1888*'
_output_shapes
:?????????@*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1888*'
_output_shapes
:?????????@2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
__inference__creator_1917
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_48*
value_dtype0	2
MutableHashTablei
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identitya
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?

?
"__inference_signature_wrapper_1652
input_1
unknown
	unknown_0	
	unknown_1:@
	unknown_2:@
	unknown_3:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_13712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?

?
*__inference_brute_force_layer_call_fn_1720
input_1
unknown
	unknown_0	
	unknown_1:@
	unknown_2:@
	unknown_3:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_15592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_1834

inputs
unknown
	unknown_0	
	unknown_1:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_14372
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1437

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_1433:@
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1433*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_13912#
!embedding/StatefulPartitionedCall?
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
|
(__inference_embedding_layer_call_fn_1885

inputs	
unknown:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_13912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
)
__inference_<lambda>_1967
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1504
queries
sequential_1486
sequential_1488	!
sequential_1490:@0
matmul_readvariableop_resource:@
gather_resource:

identity_1

identity_2??Gather?MatMul/ReadVariableOp?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_1486sequential_1488sequential_1490*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13962$
"sequential/StatefulPartitionedCall?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOp?
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identityn

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1p

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_2?
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
 __inference__traced_restore_2034
file_prefix*
assignvariableop_identifiers:/
assignvariableop_1_candidates:@9
'assignvariableop_2_embedding_embeddings:@M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: 

identity_4??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEBDquery_model/layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysBFquery_model/layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_identifiersIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_candidatesIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp'assignvariableop_2_embedding_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:3RestoreV2:tensors:4*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_23^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_3c

Identity_4IdentityIdentity_3:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_4?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_23^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"!

identity_4Identity_4:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?
?
__inference_<lambda>_19626
2key_value_init112_lookuptableimportv2_table_handle.
*key_value_init112_lookuptableimportv2_keys0
,key_value_init112_lookuptableimportv2_values	
identity??%key_value_init112/LookupTableImportV2?
%key_value_init112/LookupTableImportV2LookupTableImportV22key_value_init112_lookuptableimportv2_table_handle*key_value_init112_lookuptableimportv2_keys,key_value_init112_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init112/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init112/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2N
%key_value_init112/LookupTableImportV2%key_value_init112/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1812
input_1G
Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleH
Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value	<
*sequential_embedding_embedding_lookup_1796:@0
matmul_readvariableop_resource:@
gather_resource:

identity_1

identity_2??Gather?MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?6sequential/string_lookup/None_Lookup/LookupTableFindV2?
6sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleinput_1Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????28
6sequential/string_lookup/None_Lookup/LookupTableFindV2?
!sequential/string_lookup/IdentityIdentity?sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2#
!sequential/string_lookup/Identity?
%sequential/embedding/embedding_lookupResourceGather*sequential_embedding_embedding_lookup_1796*sequential/string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*=
_class3
1/loc:@sequential/embedding/embedding_lookup/1796*'
_output_shapes
:?????????@*
dtype02'
%sequential/embedding/embedding_lookup?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/1796*'
_output_shapes
:?????????@20
.sequential/embedding/embedding_lookup/Identity?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@22
0sequential/embedding/embedding_lookup/Identity_1?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOp?
MatMulMatMul9sequential/embedding/embedding_lookup/Identity_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identityn

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1p

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_2?
NoOpNoOp^Gather^MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup7^sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2p
6sequential/string_lookup/None_Lookup/LookupTableFindV26sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1789
input_1G
Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleH
Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value	<
*sequential_embedding_embedding_lookup_1773:@0
matmul_readvariableop_resource:@
gather_resource:

identity_1

identity_2??Gather?MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?6sequential/string_lookup/None_Lookup/LookupTableFindV2?
6sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleinput_1Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????28
6sequential/string_lookup/None_Lookup/LookupTableFindV2?
!sequential/string_lookup/IdentityIdentity?sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2#
!sequential/string_lookup/Identity?
%sequential/embedding/embedding_lookupResourceGather*sequential_embedding_embedding_lookup_1773*sequential/string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*=
_class3
1/loc:@sequential/embedding/embedding_lookup/1773*'
_output_shapes
:?????????@*
dtype02'
%sequential/embedding/embedding_lookup?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/1773*'
_output_shapes
:?????????@20
.sequential/embedding/embedding_lookup/Identity?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@22
0sequential/embedding/embedding_lookup/Identity_1?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOp?
MatMulMatMul9sequential/embedding/embedding_lookup/Identity_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identityn

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1p

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_2?
NoOpNoOp^Gather^MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup7^sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2p
6sequential/string_lookup/None_Lookup/LookupTableFindV26sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
-
__inference__initializer_1922
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
*__inference_brute_force_layer_call_fn_1703
queries
unknown
	unknown_0	
	unknown_1:@
	unknown_2:@
	unknown_3:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_15592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?

?
*__inference_brute_force_layer_call_fn_1669
input_1
unknown
	unknown_0	
	unknown_1:@
	unknown_2:@
	unknown_3:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_15042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
__inference__initializer_19076
2key_value_init112_lookuptableimportv2_table_handle.
*key_value_init112_lookuptableimportv2_keys0
,key_value_init112_lookuptableimportv2_values	
identity??%key_value_init112/LookupTableImportV2?
%key_value_init112/LookupTableImportV2LookupTableImportV22key_value_init112_lookuptableimportv2_table_handle*key_value_init112_lookuptableimportv2_keys,key_value_init112_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init112/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init112/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2N
%key_value_init112/LookupTableImportV2%key_value_init112/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?	
?
C__inference_embedding_layer_call_and_return_conditional_losses_1391

inputs	'
embedding_lookup_1385:@
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_1385inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/1385*'
_output_shapes
:?????????@*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1385*'
_output_shapes
:?????????@2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_adapt_step_1878
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*
_output_shapes
: *
output_shapes
: *
output_types
22
IteratorGetNextk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsIteratorGetNext:components:0ExpandDims/dim:output:0*
T0*
_output_shapes
:2

ExpandDimso
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsExpandDims:output:0ExpandDims_1/dim:output:0*
T0*
_output_shapes

:2
ExpandDims_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapeq
ReshapeReshapeExpandDims_1:output:0Reshape/shape:output:0*
T0*
_output_shapes
:2	
Reshape?
UniqueWithCountsUniqueWithCountsReshape:output:0*
T0*8
_output_shapes&
$:?????????::?????????*
out_idx0	2
UniqueWithCounts?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:2*
(None_lookup_table_find/LookupTableFindV2?
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2
add?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2.
,None_lookup_table_insert/LookupTableInsertV2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
9
__inference__creator_1899
identity??
hash_tabley

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name113*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
)__inference_sequential_layer_call_fn_1405
string_lookup_input
unknown
	unknown_0	
	unknown_1:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13962
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
__inference__traced_save_2012
file_prefix*
&savev2_identifiers_read_readvariableop)
%savev2_candidates_read_readvariableop3
/savev2_embedding_embeddings_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	
savev2_const_4

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEBDquery_model/layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysBFquery_model/layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_identifiers_read_readvariableop%savev2_candidates_read_readvariableop/savev2_embedding_embeddings_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1savev2_const_4"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*9
_input_shapes(
&: ::@:@::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@:

_output_shapes
::

_output_shapes
::

_output_shapes
: 
?"
?
__inference__wrapped_model_1371
input_1S
Obrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_table_handleT
Pbrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_default_value	H
6brute_force_sequential_embedding_embedding_lookup_1355:@<
*brute_force_matmul_readvariableop_resource:@)
brute_force_gather_resource:
identity

identity_1??brute_force/Gather?!brute_force/MatMul/ReadVariableOp?1brute_force/sequential/embedding/embedding_lookup?Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2?
Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Obrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_table_handleinput_1Pbrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2D
Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2?
-brute_force/sequential/string_lookup/IdentityIdentityKbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2/
-brute_force/sequential/string_lookup/Identity?
1brute_force/sequential/embedding/embedding_lookupResourceGather6brute_force_sequential_embedding_embedding_lookup_13556brute_force/sequential/string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*I
_class?
=;loc:@brute_force/sequential/embedding/embedding_lookup/1355*'
_output_shapes
:?????????@*
dtype023
1brute_force/sequential/embedding/embedding_lookup?
:brute_force/sequential/embedding/embedding_lookup/IdentityIdentity:brute_force/sequential/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*I
_class?
=;loc:@brute_force/sequential/embedding/embedding_lookup/1355*'
_output_shapes
:?????????@2<
:brute_force/sequential/embedding/embedding_lookup/Identity?
<brute_force/sequential/embedding/embedding_lookup/Identity_1IdentityCbrute_force/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@2>
<brute_force/sequential/embedding/embedding_lookup/Identity_1?
!brute_force/MatMul/ReadVariableOpReadVariableOp*brute_force_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!brute_force/MatMul/ReadVariableOp?
brute_force/MatMulMatMulEbrute_force/sequential/embedding/embedding_lookup/Identity_1:output:0)brute_force/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*
transpose_b(2
brute_force/MatMuln
brute_force/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2
brute_force/TopKV2/k?
brute_force/TopKV2TopKV2brute_force/MatMul:product:0brute_force/TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
brute_force/TopKV2?
brute_force/GatherResourceGatherbrute_force_gather_resourcebrute_force/TopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
brute_force/Gather?
brute_force/IdentityIdentitybrute_force/Gather:output:0*
T0*'
_output_shapes
:?????????
2
brute_force/Identityv
IdentityIdentitybrute_force/TopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity|

Identity_1Identitybrute_force/Identity:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1?
NoOpNoOp^brute_force/Gather"^brute_force/MatMul/ReadVariableOp2^brute_force/sequential/embedding/embedding_lookupC^brute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2(
brute_force/Gatherbrute_force/Gather2F
!brute_force/MatMul/ReadVariableOp!brute_force/MatMul/ReadVariableOp2f
1brute_force/sequential/embedding/embedding_lookup1brute_force/sequential/embedding/embedding_lookup2?
Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
+
__inference__destroyer_1912
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1559
queries
sequential_1541
sequential_1543	!
sequential_1545:@0
matmul_readvariableop_resource:@
gather_resource:

identity_1

identity_2??Gather?MatMul/ReadVariableOp?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_1541sequential_1543sequential_1545*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_14372$
"sequential/StatefulPartitionedCall?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOp?
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identityn

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1p

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_2?
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1396

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_1392:@
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1392*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_13912#
!embedding/StatefulPartitionedCall?
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_1823

inputs
unknown
	unknown_0	
	unknown_1:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13962
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
__inference_restore_fn_1954
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
+
__inference__destroyer_1927
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1766
queriesG
Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleH
Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value	<
*sequential_embedding_embedding_lookup_1750:@0
matmul_readvariableop_resource:@
gather_resource:

identity_1

identity_2??Gather?MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?6sequential/string_lookup/None_Lookup/LookupTableFindV2?
6sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handlequeriesDsequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????28
6sequential/string_lookup/None_Lookup/LookupTableFindV2?
!sequential/string_lookup/IdentityIdentity?sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2#
!sequential/string_lookup/Identity?
%sequential/embedding/embedding_lookupResourceGather*sequential_embedding_embedding_lookup_1750*sequential/string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*=
_class3
1/loc:@sequential/embedding/embedding_lookup/1750*'
_output_shapes
:?????????@*
dtype02'
%sequential/embedding/embedding_lookup?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/1750*'
_output_shapes
:?????????@20
.sequential/embedding/embedding_lookup/Identity?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@22
0sequential/embedding/embedding_lookup/Identity_1?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOp?
MatMulMatMul9sequential/embedding/embedding_lookup/Identity_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identityn

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1p

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_2?
NoOpNoOp^Gather^MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup7^sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2p
6sequential/string_lookup/None_Lookup/LookupTableFindV26sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1860

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	1
embedding_embedding_lookup_1854:@
identity??embedding/embedding_lookup?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_1854string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*2
_class(
&$loc:@embedding/embedding_lookup/1854*'
_output_shapes
:?????????@*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/1854*'
_output_shapes
:?????????@2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@2'
%embedding/embedding_lookup/Identity_1?
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^embedding/embedding_lookup,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 28
embedding/embedding_lookupembedding/embedding_lookup2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1847

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	1
embedding_embedding_lookup_1841:@
identity??embedding/embedding_lookup?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_1841string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*2
_class(
&$loc:@embedding/embedding_lookup/1841*'
_output_shapes
:?????????@*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/1841*'
_output_shapes
:?????????@2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@2'
%embedding/embedding_lookup/Identity_1?
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp^embedding/embedding_lookup,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 28
embedding/embedding_lookupembedding/embedding_lookup2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input_1,
serving_default_input_1:0?????????>
output_12
StatefulPartitionedCall_1:0?????????
>
output_22
StatefulPartitionedCall_1:1?????????
tensorflow/serving/predict:?]
?
query_model
identifiers
_identifiers

candidates
_candidates
trainable_variables
	variables
regularization_losses
	keras_api

signatures
'_default_save_signature
(__call__
*)&call_and_return_all_conditional_losses
*query_with_exclusions"
_tf_keras_model
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
trainable_variables
	variables
regularization_losses
	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_sequential
:2identifiers
:@2
candidates
'
0"
trackable_list_wrapper
5
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_metrics
metrics
trainable_variables
non_trainable_variables
	variables

layers
layer_regularization_losses
regularization_losses
(__call__
'_default_save_signature
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
,
-serving_default"
signature_map
a
lookup_table
token_counts
	keras_api
._adapt_function"
_tf_keras_layer
?

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_metrics
metrics
trainable_variables
non_trainable_variables
	variables

layers
 layer_regularization_losses
regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
&:$@2embedding/embeddings
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
1
2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
R
!_initializer
1_create_resource
2_initialize
3_destroy_resourceR 
O
4_create_resource
5_initialize
6_destroy_resourceR Z
table78
"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
"layer_metrics
#metrics
$non_trainable_variables
trainable_variables
	variables

%layers
&layer_regularization_losses
regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?B?
__inference__wrapped_model_1371input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_brute_force_layer_call_fn_1669
*__inference_brute_force_layer_call_fn_1686
*__inference_brute_force_layer_call_fn_1703
*__inference_brute_force_layer_call_fn_1720?
???
FullArgSpec/
args'?$
jself
	jqueries
jk

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_brute_force_layer_call_and_return_conditional_losses_1743
E__inference_brute_force_layer_call_and_return_conditional_losses_1766
E__inference_brute_force_layer_call_and_return_conditional_losses_1789
E__inference_brute_force_layer_call_and_return_conditional_losses_1812?
???
FullArgSpec/
args'?$
jself
	jqueries
jk

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
	jqueries
j
exclusions
jk
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_sequential_layer_call_fn_1405
)__inference_sequential_layer_call_fn_1823
)__inference_sequential_layer_call_fn_1834
)__inference_sequential_layer_call_fn_1457?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_1847
D__inference_sequential_layer_call_and_return_conditional_losses_1860
D__inference_sequential_layer_call_and_return_conditional_losses_1468
D__inference_sequential_layer_call_and_return_conditional_losses_1479?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference_signature_wrapper_1652input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_1878?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_embedding_layer_call_fn_1885?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_embedding_layer_call_and_return_conditional_losses_1894?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_1899?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_1907?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_1912?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_1917?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_1922?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_1927?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_1946checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_1954restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_35
__inference__creator_1899?

? 
? "? 5
__inference__creator_1917?

? 
? "? 7
__inference__destroyer_1912?

? 
? "? 7
__inference__destroyer_1927?

? 
? "? >
__inference__initializer_1907;<?

? 
? "? 9
__inference__initializer_1922?

? 
? "? ?
__inference__wrapped_model_1371?9,?)
"?
?
input_1?????????
? "c?`
.
output_1"?
output_1?????????

.
output_2"?
output_2?????????
Y
__inference_adapt_step_1878::0?-
&?#
!??	
? IteratorSpec
? "
 ?
E__inference_brute_force_layer_call_and_return_conditional_losses_1743?94?1
*?'
?
queries?????????

 
p 
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
E__inference_brute_force_layer_call_and_return_conditional_losses_1766?94?1
*?'
?
queries?????????

 
p
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
E__inference_brute_force_layer_call_and_return_conditional_losses_1789?94?1
*?'
?
input_1?????????

 
p 
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
E__inference_brute_force_layer_call_and_return_conditional_losses_1812?94?1
*?'
?
input_1?????????

 
p
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
*__inference_brute_force_layer_call_fn_1669|94?1
*?'
?
input_1?????????

 
p 
? "=?:
?
0?????????

?
1?????????
?
*__inference_brute_force_layer_call_fn_1686|94?1
*?'
?
queries?????????

 
p 
? "=?:
?
0?????????

?
1?????????
?
*__inference_brute_force_layer_call_fn_1703|94?1
*?'
?
queries?????????

 
p
? "=?:
?
0?????????

?
1?????????
?
*__inference_brute_force_layer_call_fn_1720|94?1
*?'
?
input_1?????????

 
p
? "=?:
?
0?????????

?
1?????????
?
C__inference_embedding_layer_call_and_return_conditional_losses_1894W+?(
!?
?
inputs?????????	
? "%?"
?
0?????????@
? v
(__inference_embedding_layer_call_fn_1885J+?(
!?
?
inputs?????????	
? "??????????@x
__inference_restore_fn_1954YK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_1946?&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
D__inference_sequential_layer_call_and_return_conditional_losses_1468n9@?=
6?3
)?&
string_lookup_input?????????
p 

 
? "%?"
?
0?????????@
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1479n9@?=
6?3
)?&
string_lookup_input?????????
p

 
? "%?"
?
0?????????@
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1847a93?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????@
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1860a93?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????@
? ?
)__inference_sequential_layer_call_fn_1405a9@?=
6?3
)?&
string_lookup_input?????????
p 

 
? "??????????@?
)__inference_sequential_layer_call_fn_1457a9@?=
6?3
)?&
string_lookup_input?????????
p

 
? "??????????@?
)__inference_sequential_layer_call_fn_1823T93?0
)?&
?
inputs?????????
p 

 
? "??????????@?
)__inference_sequential_layer_call_fn_1834T93?0
)?&
?
inputs?????????
p

 
? "??????????@?
"__inference_signature_wrapper_1652?97?4
? 
-?*
(
input_1?
input_1?????????"c?`
.
output_1"?
output_1?????????

.
output_2"?
output_2?????????
